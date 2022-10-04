# # Train enformer
# Eran Mukamel
# emukamel@ucsd.edu

from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter,Namespace

################
# Parse command line inputs
parser = ArgumentParser(
    description="""
    Train an enformer model with side-inputs.
    Eran Mukamel (emukamel@ucsd.edu)
    """,
    formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--predictors_dirs',default='', type=str, 
                    help='Directory (or list of directories) containing the predictors TF dataset.')
parser.add_argument('--targets_dir',default='', type=str,
                    help='Directory containing targets TF dataset')
parser.add_argument('--run_id',default='enformer_MLP', type=str,
                    help='Prefix for output files (checkpoints)')
parser.add_argument('--log_target',default='False',type=str,
                    help='Whether or not to take the log of the target')
parser.add_argument('--use_sequence',default='True', type=str,
                    help='Whether to use sequence as a predictor')
parser.add_argument('--only_sequence',default='False', type=str,
                    help='Whether to use ONLY the sequence as a predictor, ignoring other epigenetic tracks. (Overrides use_sequence).')


parser.add_argument('--out_activation',default='sigmoid', type=str,choices=['sigmoid','softplus','linear','relu'],
                    help='Output activation function')
parser.add_argument('--num_heads',default=8, type=int,
                    help='Number of heads for internal transformer')
parser.add_argument('--nchannels',default=1536 // 4, type=int,
                    help='Width of network (num. channels)')
parser.add_argument('--num_transformer_layers',default=11, type=int,
                    help='Number of transformer layers')
parser.add_argument('--pooling_type',default='max', type=str,
                    help='Type of pooling for internal layers')
parser.add_argument('--model_architecture',default=False,type=str,choices=['EpiEnformer_SideTrunk','EpiEnformer_TwoStems'],
                    help='''
  Which model architecture to use. This defines the class that will be used from the module "enformer_epiAttend." 
    * EpiEnformer_SideTrunk - add the epigenetic data as a side input after the MHA layers
    * EpiEnformer_TwoStems - add epigenetic data before the MHA layers''')

group_learn = parser.add_argument_group('Learning parameters:')
group_learn.add_argument('--num_warmup_steps',default=5000, type=int,
                    help='Number of warmup learning steps during which learning rate linearly ramps up to target rate.')
group_learn.add_argument('--target_learning_rate',default=.0005, type=float,
                    help='Target learning rate')
group_learn.add_argument('--loss',default='mse', type=str,choices=['mse','poisson'],
                    help='Loss function')
group_learn.add_argument('--epochs',default=5000, type=int,
                    help='Number of training epochs')
group_learn.add_argument('--validation_interval',default=30, type=float,
                    help='Minimum time (seconds) between validation and checkpoint saves')
group_learn.add_argument('--validation_steps',default=100, type=float,
                    help='Max number of steps between validation and checkpoint saves')
group_learn.add_argument('--lr_adjust_factor',default=1,type=float,
                    help='Factor by which to multiply learning rate if the validation loss is not decreasing')
group_learn.add_argument('--lr_adjust_patience',default=10,type=int,
                    help='Patience for learning rate adjustment')

group_learn.add_argument('--restart_ckpt',default=None,type=str,
                    help='Name of checkpoint file to use as starting point for initializing model. NOTE: If restarting, the stored arguments will override all the other inputs (except run_id)')

group_st = parser.add_argument_group('Parameters for the EpiEnformer_SideTrunk architecture:')
group_st.add_argument('--side_trunk_depth',default=1,type=int,
                    help='Number of layers of convolution for the side trunk.')

args=parser.parse_args()

args.use_sequence = args.use_sequence.lower()!='false'
args.only_sequence = args.only_sequence.lower()!='false'
args.log_target = args.log_target.lower()!='false'
args.predictors_dirs = args.predictors_dirs.split(',')

################

import time,datetime,os,glob,json
import socket
basedir='/tuba/datasets/enformer'

today=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if args.restart_ckpt is not None:
  with open(f'{args.restart_ckpt}/arguments.txt','r') as file:
    args_stored=json.load(file)
  args_new=vars(args)
  args_stored['run_id'] = args_new['run_id']
  args_stored['restart_ckpt'] = args_new['restart_ckpt']
  for i in args_new:
    if args_stored[i]!=args_new[i]:
      print(f'*** Replacing {i}: Old={args_stored[i]}, New={args_new[i]}')
  args=Namespace(**args_stored)

ckpt_dir = f'{basedir}/checkpoints/{args.run_id}_{today}'
nckpts = len(glob.glob(ckpt_dir+'-*'))
ckpt_dir = f'{ckpt_dir}-{nckpts}'
os.mkdir(ckpt_dir)

with open(f'{ckpt_dir}/arguments.txt','w') as file:
    json.dump(vars(args), file, indent=4)
print(f'*** Saving checkpoints to: {ckpt_dir}')

#######

import tensorflow as tf
# Make sure the GPU is enabled 
assert tf.config.list_physical_devices('GPU'), 'No GPU available??'

import sonnet as snt
from tqdm import tqdm
import numpy as np
import pandas as pd

import enformer_epiAttend as enf
import my_enformer as me

datasets={}
for t in ['train','valid']:
  predictors_datasets={}
  if args.only_sequence:  
    predictors_datasets['sequence']=me.get_dataset(args.predictors_dirs[0], t).map(lambda x: x['sequence'])
  else:
    if args.use_sequence:
      predictors_datasets['sequence']=me.get_dataset(args.predictors_dirs[0], t).map(lambda x: x['sequence'])
    for predictor_dir in args.predictors_dirs:
      print(predictor_dir)
      predictors_datasets[predictor_dir]=me.get_dataset_targets(predictor_dir, t)
  predictors_datasets = tf.data.Dataset.zip(tuple(predictors_datasets.values()))
  targets_dataset = me.get_dataset_targets(args.targets_dir, t)

  if args.log_target:
    datasets[t] = tf.data.Dataset.zip((predictors_datasets,targets_dataset)).map(lambda x,y:{'predictors':x,'target':tf.math.log(1+tf.cast(y,tf.float32))/tf.math.log(10.)})
  else:
    datasets[t] = tf.data.Dataset.zip((predictors_datasets,targets_dataset)).map(lambda x,y:{'predictors':x,'target':y})
  datasets[t] = datasets[t].batch(1).repeat().prefetch(2)

# # Get track names
# mc_track_metadata = pd.read_csv('../prepare_training_data/CEMBA_mc_mincov5_smooth8_cg_ch/targets.txt',sep='\t',index_col=0)
# mc_track_metadata['celltype']=mc_track_metadata.description.str.extract('(.*)_c[gh]')
# mc_track_metadata['mc_type']=mc_track_metadata.description.str.extract('(c[gh])')
# atac_track_metadata = pd.read_csv('../prepare_training_data/Li2021_mouse_snATAC_binsize128_BasenjiBins/targets.txt',sep='\t',index_col=0)

batch_mc=next(iter(datasets['train']))

learning_rate = tf.Variable(0., trainable=False, name='learning_rate')
optimizer = snt.optimizers.Adam(learning_rate=learning_rate)
num_warmup_steps = args.num_warmup_steps
target_learning_rate = args.target_learning_rate

heads_channels = {'target': batch_mc['target'].shape[-1]}

model = getattr(enf,args.model_architecture)(channels=args.nchannels,  # Use 4x fewer channels to train faster.
                          num_heads=args.num_heads, # 8
                          num_transformer_layers=args.num_transformer_layers, # 11
                          heads_channels = heads_channels,
                          pooling_type=args.pooling_type, # max
                          out_activation=args.out_activation,
                         )

def create_step_function(model, optimizer):
  @tf.function
  def train_step(batch, head, optimizer_clip_norm_global=0.2):
    with tf.GradientTape() as tape:
      outputs = model(batch['predictors'], is_training=True)[head]
      if args.loss=='poisson':
          loss = tf.reduce_mean(
              tf.keras.losses.poisson(batch['target'], outputs)
          )
      else:
          loss = tf.reduce_mean(
              tf.keras.losses.mse(batch['target'], outputs)
          )

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply(gradients, model.trainable_variables)

    return loss
  return train_step

train_step = create_step_function(model, optimizer)

#########
# Run the training
steps_per_epoch = 20

data_it=iter(datasets['train'])
global_step_tf = tf.Variable(1.0)
global_step=global_step_tf.numpy()
ckpt = tf.train.Checkpoint(step=global_step_tf, 
                           optimizer=optimizer, net=model, iterator=data_it)
if args.restart_ckpt is not None:
  manager = tf.train.CheckpointManager(ckpt,args.restart_ckpt,max_to_keep=3)
  ckpt.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print(" >>>> Restored model weights from: {}".format(manager.latest_checkpoint))
  else:
    print(" >>>> Initializing model weights from scratch.")

manager = tf.train.CheckpointManager(ckpt,ckpt_dir,max_to_keep=3)
print(f' >>>> Creating checkpoint at {ckpt_dir}')


# Logging -- allows monitoring with tensorboard
train_log_dir = f'{basedir}/logs/{args.run_id}_{today}/train'
test_log_dir =  f'{basedir}/logs/{args.run_id}_{today}/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)
print(f'+++ Writing logs to: {train_log_dir}')

# Adjust learning rate
lr_adjust={'factor':args.lr_adjust_factor, 'patience':args.lr_adjust_patience, 'counter':0}

losses = [tf.keras.losses.mse, tf.keras.losses.poisson]
history = {'global_step':[],
           'loss_valid_mse':[],
           'loss_valid_poisson':[],
           'learning_rate':[]}

t0=time.time()-100
nvalid=8
for epoch_i in range(args.epochs):
  for i in tqdm(range(steps_per_epoch),
                desc=f'global_step={global_step}, elapsed t={(time.time()-t0):3.2f}s'):
    global_step_tf.assign_add(1.)
    global_step=global_step_tf.numpy()

    if global_step > 1:
      learning_rate_frac = tf.math.minimum(
          1.0, global_step / tf.math.maximum(1.0, num_warmup_steps))      
      learning_rate.assign(target_learning_rate * learning_rate_frac)

    batch_mc=next(data_it)
    
    loss_train = train_step(batch=batch_mc, head='target')
    ckpt.step.assign_add(1)

    with train_summary_writer.as_default():
      tf.summary.scalar(f'loss_{args.loss}', loss_train, step=global_step)
      tf.summary.scalar('learning_rate', learning_rate.numpy(), step=global_step)

  if ((time.time()-t0)>args.validation_interval) or ((global_step-np.max(history['global_step']))>args.validation_steps):
        t0=time.time()

        data_it_valid=iter(datasets['valid'])
        loss_valid={'mse':[],'poisson':[]}
        for i in tqdm(range(nvalid),desc='Validation'):
            batch_mc_valid=next(data_it_valid)
            outputs = model(batch_mc_valid['predictors'], is_training=False)['target']
            for loss,loss_type in zip(losses,loss_valid.keys()):
              loss_valid[loss_type].append(tf.reduce_mean(loss(batch_mc_valid['target'], outputs)).numpy())
        for loss_type in loss_valid.keys():
          loss_valid[loss_type]=np.mean(loss_valid[loss_type])
            
        history['global_step'].append(global_step)
        for l in ['mse','poisson']:
          history[f'loss_valid_{l}'].append(loss_valid[l])
        history['learning_rate'].append(learning_rate.numpy())

        # End of epoch.
        print('')
        print(
            'loss_valid - mse', loss_valid['mse'],
            'loss_valid - poisson', loss_valid['poisson'],
            'learning_rate', optimizer.learning_rate.numpy()
            )

        # Save checkpoint
        min_loss=np.min(history['loss_valid_'+args.loss])
        if loss_valid[args.loss]==min_loss:
            save_path = manager.save()
            print(f'++++ Saved checkpoint: {save_path}')
            lr_adjust['counter']=0
        else:
            print(f'++++ Validation loss increased, not saving checkpoint. Min loss={min_loss}')
            lr_adjust['counter']+=1
        if lr_adjust['counter']>lr_adjust['patience']:
          target_learning_rate *= lr_adjust['factor']
          lr_adjust['counter']=0
        with test_summary_writer.as_default():
          for l in ['mse','poisson']:
            tf.summary.scalar(f'loss_{l}', loss_valid[l], step=global_step)
          tf.summary.scalar('learning_rate', learning_rate.numpy(), step=global_step)
      
        pd.DataFrame.from_dict(history).to_csv(f'{ckpt_dir}/training_history.csv')
