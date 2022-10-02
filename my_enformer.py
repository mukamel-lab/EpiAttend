import tensorflow as tf
import sonnet as snt
import numpy as np
import pandas as pd
import time,os,re
import matplotlib.pyplot as plt
import json
import functools
from multiprocessing import Pool
import tensorflow_hub as hub
from kipoiseq import Interval
import kipoiseq
import pyfaidx

node=os.uname().nodename
print(f'**** Running on {node}')
if re.match(r'^ssrde',node):
    enformer_dir = '/home/emukamel/emukamel/enformer'
else:
    enformer_dir = '/cndd/emukamel/enformer'

def get_targets(tfdata_dir):
    targets_txt=f'{enformer_dir}/prepare_training_data/{tfdata_dir}/targets.txt'
    return pd.read_csv(targets_txt, sep='\t')


def organism_path(organism, tfdata_dir=0):
    if ((organism=='mouse') or (organism=='human')):
        return os.path.join('gs://basenji_barnyard/data', organism)
    else:
        if tfdata_dir==0:
            tfdata_dir=organism
        # TODO: Make this more general
        if not os.path.isabs(tfdata_dir):
            tfdata_dir = f'{enformer_dir}/prepare_training_data/{tfdata_dir}'
        return os.path.join(tfdata_dir)        

# Helper routines for loading data
def get_dataset(organism, subset, tfdata_dir=0, num_threads=8):
  metadata = get_metadata(organism, tfdata_dir=tfdata_dir)
  dataset = tf.data.TFRecordDataset(tfrecord_files(organism, subset, tfdata_dir=tfdata_dir),
                                    compression_type='ZLIB',
                                    num_parallel_reads=num_threads,
                                   )
  dataset = dataset.map(functools.partial(deserialize, metadata=metadata),
                        num_parallel_calls=num_threads,
                       )
    
  return dataset


def get_metadata(organism, tfdata_dir=0):
  # Keys:
  # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
  # pool_width, crop_bp, target_length
  path = os.path.join(organism_path(organism, tfdata_dir=tfdata_dir), 'statistics.json')
  with tf.io.gfile.GFile(path, 'r') as f:
    return json.load(f)


def tfrecord_files(organism, subset, tfdata_dir=0):
  # Sort the values by int(*).
  return sorted(tf.io.gfile.glob(os.path.join(
      organism_path(organism, tfdata_dir=tfdata_dir), 'tfrecords', f'{subset}-*.tfr'
  )), key=lambda x: int(x.split('-')[-1].split('.')[0]))


def deserialize(serialized_example, metadata):
  """Deserialize bytes stored in TFRecordFile."""
  feature_map = {
      'sequence': tf.io.FixedLenFeature([], tf.string),
      'target': tf.io.FixedLenFeature([], tf.string),
  }
  example = tf.io.parse_example(serialized_example, feature_map)
  sequence = tf.io.decode_raw(example['sequence'], tf.bool)
  sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
  sequence = tf.cast(sequence, tf.float32)
  target = tf.io.decode_raw(example['target'], tf.float16)
  target = tf.reshape(target,
                      (metadata['target_length'], metadata['num_targets']))
  target = tf.cast(target, tf.float32)
  return {'sequence': sequence,'target': target}

## Model training
def create_step_function(model, optimizer):
  heads=model.heads.keys()

  @tf.function
  def train_step(batch, heads=heads, optimizer_clip_norm_global=0.2):
    losses={}
    for head in heads:
        with tf.GradientTape() as tape:
          outputs = model(batch[head]['sequence'], is_training=True)[head]
          if head=='mouse_mcg':
                # EAM Oct. 2021: Use binary cross-entropy for mCG data
                # TODO: Exclude sequences with too many zeros...
                tgt = batch[head]['target']
                tgt = tf.keras.backend.clip(tgt, 0.5,1)
                loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(tgt, outputs))
          else:
              loss = tf.reduce_mean(
                  tf.keras.losses.poisson(batch[head]['target'], outputs))

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply(gradients, model.trainable_variables)
        losses['loss_'+head] = loss

    return losses
  return train_step

# @title `Enformer`, `EnformerScoreVariantsNormalized`, `EnformerScoreVariantsPCANormalized`,
# SEQUENCE_LENGTH = 393216

class Enformer:

  def __init__(self, tfhub_url):
    self._model = hub.load(tfhub_url).model

  def predict_on_batch(self, inputs):
    predictions = self._model.predict_on_batch(inputs)
    return {k: v.numpy() for k, v in predictions.items()}

  @tf.function
  def contribution_input_grad(self, input_sequence,
                              target_mask, output_head='human'):
    input_sequence = input_sequence[tf.newaxis]

    target_mask_mass = tf.reduce_sum(target_mask)
    with tf.GradientTape() as tape:
      tape.watch(input_sequence)
      prediction = tf.reduce_sum(
          target_mask[tf.newaxis] *
          self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

    input_grad = tape.gradient(prediction, input_sequence) * input_sequence
    input_grad = tf.squeeze(input_grad, axis=0)
    return tf.reduce_sum(input_grad, axis=-1)


class EnformerScoreVariantsRaw:

  def __init__(self, tfhub_url, organism='human'):
    self._model = Enformer(tfhub_url)
    self._organism = organism

  def predict_on_batch(self, inputs):
    ref_prediction = self._model.predict_on_batch(inputs['ref'])[self._organism]
    alt_prediction = self._model.predict_on_batch(inputs['alt'])[self._organism]

    return alt_prediction.mean(axis=1) - ref_prediction.mean(axis=1)


class EnformerScoreVariantsNormalized:

  def __init__(self, tfhub_url, transform_pkl_path,
               organism='human'):
    assert organism == 'human', 'Transforms only compatible with organism=human'
    self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
    with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
      transform_pipeline = joblib.load(f)
    self._transform = transform_pipeline.steps[0][1]  # StandardScaler.

  def predict_on_batch(self, inputs):
    scores = self._model.predict_on_batch(inputs)
    return self._transform.transform(scores)


class EnformerScoreVariantsPCANormalized:

  def __init__(self, tfhub_url, transform_pkl_path,
               organism='human', num_top_features=500):
    self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
    with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
      self._transform = joblib.load(f)
    self._num_top_features = num_top_features

  def predict_on_batch(self, inputs):
    scores = self._model.predict_on_batch(inputs)
    return self._transform.transform(scores)[:, :self._num_top_features]

def _reduced_shape(shape, axis):
  if axis is None:
    return tf.TensorShape([])
  return tf.TensorShape([d for i, d in enumerate(shape) if i not in axis])


class CorrelationStats(tf.keras.metrics.Metric):
  """Contains shared code for PearsonR and R2."""

  def __init__(self, reduce_axis=None, name='pearsonr'):
    """Pearson correlation coefficient.

    Args:
      reduce_axis: Specifies over which axis to compute the correlation (say
        (0, 1). If not specified, it will compute the correlation across the
        whole tensor.
      name: Metric name.
    """
    super(CorrelationStats, self).__init__(name=name)
    self._reduce_axis = reduce_axis
    self._shape = None  # Specified in _initialize.

  def _initialize(self, input_shape):
    # Remaining dimensions after reducing over self._reduce_axis.
    self._shape = _reduced_shape(input_shape, self._reduce_axis)

    weight_kwargs = dict(shape=self._shape, initializer='zeros')
    self._count = self.add_weight(name='count', **weight_kwargs)
    self._product_sum = self.add_weight(name='product_sum', **weight_kwargs)
    self._true_sum = self.add_weight(name='true_sum', **weight_kwargs)
    self._true_squared_sum = self.add_weight(name='true_squared_sum',
                                             **weight_kwargs)
    self._pred_sum = self.add_weight(name='pred_sum', **weight_kwargs)
    self._pred_squared_sum = self.add_weight(name='pred_squared_sum',
                                             **weight_kwargs)

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Update the metric state.

    Args:
      y_true: Multi-dimensional float tensor [batch, ...] containing the ground
        truth values.
      y_pred: float tensor with the same shape as y_true containing predicted
        values.
      sample_weight: 1D tensor aligned with y_true batch dimension specifying
        the weight of individual observations.
    """
    if self._shape is None:
      # Explicit initialization check.
      self._initialize(y_true.shape)
    y_true.shape.assert_is_compatible_with(y_pred.shape)
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    self._product_sum.assign_add(
        tf.reduce_sum(y_true * y_pred, axis=self._reduce_axis))

    self._true_sum.assign_add(
        tf.reduce_sum(y_true, axis=self._reduce_axis))

    self._true_squared_sum.assign_add(
        tf.reduce_sum(tf.math.square(y_true), axis=self._reduce_axis))

    self._pred_sum.assign_add(
        tf.reduce_sum(y_pred, axis=self._reduce_axis))

    self._pred_squared_sum.assign_add(
        tf.reduce_sum(tf.math.square(y_pred), axis=self._reduce_axis))

    self._count.assign_add(
        tf.reduce_sum(tf.ones_like(y_true), axis=self._reduce_axis))

  def result(self):
    raise NotImplementedError('Must be implemented in subclasses.')

  def reset_states(self):
    if self._shape is not None:
      tf.keras.backend.batch_set_value([(v, np.zeros(self._shape))
                                        for v in self.variables])


class PearsonR(CorrelationStats):
  """Pearson correlation coefficient.

  Computed as:
  ((x - x_avg) * (y - y_avg) / sqrt(Var[x] * Var[y])
  """

  def __init__(self, reduce_axis=(0,), name='pearsonr'):
    """Pearson correlation coefficient.

    Args:
      reduce_axis: Specifies over which axis to compute the correlation.
      name: Metric name.
    """
    super(PearsonR, self).__init__(reduce_axis=reduce_axis,
                                   name=name)

  def result(self):
    true_mean = self._true_sum / self._count
    pred_mean = self._pred_sum / self._count

    covariance = (self._product_sum
                  - true_mean * self._pred_sum
                  - pred_mean * self._true_sum
                  + self._count * true_mean * pred_mean)

    true_var = self._true_squared_sum - self._count * tf.math.square(true_mean)
    pred_var = self._pred_squared_sum - self._count * tf.math.square(pred_mean)
    tp_var = tf.math.sqrt(true_var) * tf.math.sqrt(pred_var)
    correlation = covariance / tp_var

    return correlation


class R2(CorrelationStats):
  """R-squared  (fraction of explained variance)."""

  def __init__(self, reduce_axis=None, name='R2'):
    """R-squared metric.

    Args:
      reduce_axis: Specifies over which axis to compute the correlation.
      name: Metric name.
    """
    super(R2, self).__init__(reduce_axis=reduce_axis,
                             name=name)

  def result(self):
    true_mean = self._true_sum / self._count
    total = self._true_squared_sum - self._count * tf.math.square(true_mean)
    residuals = (self._pred_squared_sum - 2 * self._product_sum
                 + self._true_squared_sum)

    return tf.ones_like(residuals) - residuals / total

class MetricDict:
  def __init__(self, metrics):
    self._metrics = metrics

  def update_state(self, y_true, y_pred):
    for k, metric in self._metrics.items():
      metric.update_state(y_true, y_pred)

  def result(self):
    return {k: metric.result() for k, metric in self._metrics.items()}

def evaluate_model(model, dataset, head, max_steps=None):
  metric = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1))})
  @tf.function
  def predict(x):
    return model(x, is_training=False)[head]

  for i, batch in enumerate(dataset):
    if max_steps is not None and i > max_steps:
      break
    metric.update_state(batch['target'], predict(batch['sequence']))

  return metric.result()
# @title `variant_centered_sequences`

class FastaStringExtractor:
    
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


def variant_generator(vcf_file, gzipped=False):
  """Yields a kipoiseq.dataclasses.Variant for each row in VCF file."""
  def _open(file):
    return gzip.open(vcf_file, 'rt') if gzipped else open(vcf_file)
    
  with _open(vcf_file) as f:
    for line in f:
      if line.startswith('#'):
        continue
      chrom, pos, id, ref, alt_list = line.split('\t')[:5]
      # Split ALT alleles and return individual variants as output.
      for alt in alt_list.split(','):
        yield kipoiseq.dataclasses.Variant(chrom=chrom, pos=pos,
                                           ref=ref, alt=alt, id=id)


def one_hot_encode(sequence):
  return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def variant_centered_sequences(vcf_file, sequence_length, gzipped=False,
                               chr_prefix=''):
  seq_extractor = kipoiseq.extractors.VariantSeqExtractor(
    reference_sequence=FastaStringExtractor(fasta_file))

  for variant in variant_generator(vcf_file, gzipped=gzipped):
    interval = Interval(chr_prefix + variant.chrom,
                        variant.pos, variant.pos)
    interval = interval.resize(sequence_length)
    center = interval.center() - interval.start

    reference = seq_extractor.extract(interval, [], anchor=center)
    alternate = seq_extractor.extract(interval, [variant], anchor=center)

    yield {'inputs': {'ref': one_hot_encode(reference),
                      'alt': one_hot_encode(alternate)},
           'metadata': {'chrom': chr_prefix + variant.chrom,
                        'pos': variant.pos,
                        'id': variant.id,
                        'ref': variant.ref,
                        'alt': variant.alt}}

def smooth(signal, smf=10):
    if smf==1:
        return signal
    try:
        signal = signal.numpy()
    except:
        pass
    signal_sm = []
    kernel = np.ones(smf)/smf
    if signal.ndim==1:
        signal_sm = np.convolve(signal,kernel,'same')
        signal_sm[:smf] = np.nan
        signal_sm[-smf:] = np.nan
    else:
        for i in range(signal.shape[1]):
            su=signal[:,i].flatten()
            su=np.convolve(su,kernel,'same')
            signal_sm.append(su)
        signal_sm = np.stack(signal_sm).T
    return signal_sm

#################################################
##### Dataset loading routines for my data -- EAM
def get_dataset_targets(organism, subset, tfdata_dir=0, num_threads=8):
  # EAM: Return only the targets (not sequences)
  metadata = get_metadata(organism, tfdata_dir=tfdata_dir)
  dataset = tf.data.TFRecordDataset(tfrecord_files(organism, subset, tfdata_dir=tfdata_dir),
                                    compression_type='ZLIB',
                                    num_parallel_reads=num_threads)
  dataset = dataset.map(functools.partial(deserialize_targets, metadata=metadata),
                        num_parallel_calls=num_threads)
  return dataset

def deserialize_targets(serialized_example, metadata):
  """Deserialize bytes stored in TFRecordFile; EAM: Keep only target, not sequence."""
  feature_map = {
      'sequence': tf.io.FixedLenFeature([], tf.string),
      'target': tf.io.FixedLenFeature([], tf.string),
  }
  example = tf.io.parse_example(serialized_example, feature_map)
  target = tf.io.decode_raw(example['target'], tf.float16)
  target = tf.reshape(target,
                      (metadata['target_length'], metadata['num_targets']))
  # TODO: Filter out training samples that have little or no variance
  target = tf.cast(target, tf.float32)
  return target

def get_multi_datasets(predictors_dirs, targets_dir, only_sequence=False, use_sequence=True):
  datasets={}

  # Make sure flags are boolean
  if type(use_sequence) is str:
    use_sequence = use_sequence.lower()!='false'
  if type(only_sequence) is str:
    only_sequence = only_sequence.lower()!='false'

  if type(predictors_dirs) is not list:
      predictors_dirs = predictors_dirs.split(',')
  for t in ['train','valid']:
    predictors_datasets={}
    if only_sequence:  
      predictors_datasets['sequence']=get_dataset(predictors_dirs[0], t).map(lambda x: x['sequence'])
    else:
      if use_sequence:
        predictors_datasets['sequence']=get_dataset(predictors_dirs[0], t).map(lambda x: x['sequence'])
      for predictor_dir in predictors_dirs:
        predictors_datasets[predictor_dir]=get_dataset_targets(predictor_dir, t)
    predictors_datasets = tf.data.Dataset.zip(tuple(predictors_datasets.values()))
    targets_dataset = get_dataset_targets(targets_dir, t)

    datasets[t] = tf.data.Dataset.zip((predictors_datasets,targets_dataset)).map(lambda x,y:{'predictors':x,'target':y})
    datasets[t] = datasets[t].batch(1).repeat().prefetch(2)
  return datasets

### Plotting functions -- EAM
def plot_training(df_training):
    fig,axs = plt.subplots(2,1)
    df_training.plot(kind='scatter', x='global_step',y='valid_pearson_r_mouse_mcg',ax=axs[0])
    # df_training.plot(kind='scatter', x='global_step',y='loss_mouse_mcg')
    try:
        df_training['loss_mouse_mcg_sm'] = smooth(df_training['loss_mouse_mcg'])
        # df_training.loc[:10, 'loss_mouse_mcg_sm'] = np.nan
        # df_training.loc[-10:, 'loss_mouse_mcg_sm'] = np.nan
        df_training.plot(kind='line', x='global_step',y=['loss_mouse_mcg','loss_mouse_mcg_sm'],ax=axs[1])
        # df_training.loc[50:100,'loss_mouse_mcg_sm']
    except:
        print('No smoothing...')

# def get_one_chunk(model,dataset,smf,head):
#         batch = next(dataset)
#         mcg1 = smooth(batch['target'].numpy().squeeze(), smf=smf)
#         mcg_hat1 = smooth(model(batch['sequence'],is_training=False)[head].numpy().squeeze(),smf=smf)
#         return {'mcg':mcg1, 'mcg_hat':mcg_hat1}
    
def get_prediction(model, nchunks=10, batchsize=1, smf=1, head='mouse_mcg', datasets=None, tfdata_dir=0,
                   subset='train'
                  ):
    # Visualize predicted mCG level for some example validation regions
    if datasets==None:
        dataset=iter(get_dataset(head,subset,tfdata_dir=tfdata_dir).batch(batchsize).prefetch(2))
    else:
        dataset=datasets[head]

    # I don't think this can be parallelized using multiprocessing
    mcg, mcg_hat = [],[]
    for chunk in range(nchunks):
        batch = next(dataset)
        mcg1 = smooth(batch['target'].numpy().squeeze(), smf=smf)
        mcg.append(mcg1)
        mcg_hat1 = smooth(model(batch['sequence'],is_training=False)[head].numpy().squeeze(),smf=smf)
        mcg_hat.append(mcg_hat1)
    mcg = np.concatenate(mcg,axis=0)
    mcg_hat = np.concatenate(mcg_hat,axis=0)
    
    ncells = mcg.shape[-1]
    mcg = np.reshape(mcg, [-1,ncells])
    mcg_hat = np.reshape(mcg_hat, [-1,ncells])
    
    return mcg,mcg_hat

def plot_predictions(mcg, mcg_hat, smf=10,yspace=1,
                     subtract_mean=False,
                     zscore=False,
                     xstart=None,xstop=None,
                     dmrs=np.array([]),
                     tracks=-1,
                    ):
    binsize=128
    ncelltypes = mcg.shape[1]
    if (tracks==-1):
        tracks = np.arange(ncelltypes)
    else:
        mcg = mcg[:,tracks]
        mcg_hat = mcg_hat[:,tracks]
        ncelltypes = mcg.shape[1]
    gvec = np.arange(mcg.shape[0])*binsize
    fig=plt.figure(figsize=(25,0.8*ncelltypes))
    mcg_show, mcg_hat_show = mcg, mcg_hat
    if xstart is None:
        xstart=gvec[0]
    if xstop is None:
        xstop=gvec[-1]
    if smf>1:
        mcg_show = smooth(mcg_show,smf=smf)
        mcg_hat_show = smooth(mcg_hat_show,smf=smf)
    
    guse = (gvec>=xstart) & (gvec<=xstop)
    mcg_show = mcg_show[guse,:]
    mcg_hat_show = mcg_hat_show[guse,:]
    dmrs = dmrs[(dmrs>=xstart) & (dmrs<=xstop)]
    gvec = gvec[guse]
    
    if subtract_mean:
        mcg_show = mcg_show - np.median(mcg_show,axis=1,keepdims=True)
        mcg_hat_show = mcg_hat_show - np.median(mcg_hat_show,axis=1,keepdims=True)
    if zscore:
        mcg_show = mcg_show - np.mean(mcg_show,axis=0,keepdims=True)
        mcg_show = mcg_show / np.std(mcg_show,axis=0,keepdims=True)
        mcg_hat_show = mcg_hat_show - np.mean(mcg_hat_show,axis=0,keepdims=True)
        mcg_hat_show = mcg_hat_show / np.std(mcg_hat_show,axis=0,keepdims=True)
#         yspace = yspace/2
        
#     df = pd.DataFrame.from_dict({'Position':gvec})
#     for i in range(ncelltypes):
#         df['mCG C%d'%i] = mcg_show[:,i]
#     return df
    l=plt.plot(gvec,mcg_show+yspace*np.arange(ncelltypes),
             color='k')
    l[0].set_label('Data')
    l=plt.plot(gvec,mcg_hat_show+yspace*np.arange(ncelltypes))
    l[0].set_label('Predition')
    plt.legend()
    plt.xlabel('Genomic position (bp)')
    plt.ylabel('mCG')
    plt.yticks(ticks=yspace*np.arange(ncelltypes), labels=[f'Cluster{d}' for d in range(ncelltypes)])
    
    plt.vlines(dmrs, *plt.ylim(), 'gray')
    
    plt.title(f'{binsize}bp bins')
    plt.axis('tight')
    return fig

def mcg_histplot(mcg,mcg_hat,nbins=50):
    """Make a joint histogram plot of mcg vs. mcg_hat"""
    import seaborn as sns
    from scipy.stats import spearmanr
    g=sns.jointplot(x=mcg.flatten(),y=mcg_hat.flatten(),
                    vmin=0,vmax=2000,
                    cmap='Reds',
                    kind='hist',
                    bins=[np.linspace(0,1,nbins),np.linspace(0,1,nbins)],
                    cbar=True,
                    cbar_kws={'shrink':0.5,'label':'Bins','ticks':[0,1000,2000]})
    g.ax_joint.set_xlabel('mCG - data')
    g.ax_joint.set_ylabel('mCG - predicted')

    g.ax_joint.set_xlim([0,1])
    g.ax_joint.set_ylim([0,1])
    g.ax_marg_y.set_xscale('log')
    g.ax_marg_x.set_yscale('log')
    g.fig.get_axes()[-1].set_position([.9,.1,.2,.1])

    cc = spearmanr(mcg, mcg_hat).correlation
    ncells = mcg.shape[1]
    cc = cc[:ncells,ncells:]
    cc_mean,cc_std = np.mean(cc),np.std(cc)
    
    cc = np.corrcoef(mcg.T, mcg_hat.T)
    cc = cc[:ncells,ncells:]
    cc_pearson_mean,cc_pearson_std = np.mean(cc),np.std(cc)
    
    g.ax_marg_x.set_title('Spearman r=%3.3f±%3.3f, Pearson r=%3.3f±%3.3f' % (cc_mean,cc_std,cc_pearson_mean,cc_pearson_std))
    
    
    
    return g

def mcg_corrplot(mcg,mcg_hat,show_all=False,vmin=0,vmax=1,corrtype='spearman'):
    import seaborn as sns
    if corrtype=='spearman':
        from scipy.stats import spearmanr
        cc = spearmanr(mcg, mcg_hat).correlation
    elif corrtype=='pearson':
        cc = np.corrcoef(mcg.T, mcg_hat.T)
    else:
        raise Exception("corrtype must be 'spearman' or 'pearson'")
        
    ncells = mcg.shape[1]
    if not show_all:
        cc = cc[:ncells,ncells:]
    plt.figure()
    sns.heatmap(cc,vmin=vmin,vmax=vmax)
    plt.title('r = %3.3f ± %3.3f' % (cc.mean(), cc.std()))
    return cc
