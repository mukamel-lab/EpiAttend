# Helper functions for plotting
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

import sonnet as snt
from tqdm import tqdm
import numpy as np
import pandas as pd
import time,os

def load_args(checkpoint):
    # Load the arguments that were used for training
    with open(f'{checkpoint}/arguments.txt','r') as file:
        args=json.load(file)
    if not 'side_trunk_depth' in args:
        args['side_trunk_depth']=1
        if 'deep' in checkpoint:
            args['side_trunk_depth']=3
    if not 'model_architecture' in args:
        args['model_architecture']='EpiEnformer_SideTrunk'
        
    return args
    
def restore_checkpoint(checkpoint):
    args=load_args(checkpoint)
    datasets=me.get_multi_datasets(args['predictors_dirs'], args['targets_dir'], 
      args['only_sequence'], args['use_sequence'])
    data_it=iter(datasets['train'])
    batch=next(data_it)
    model = getattr(enf,args['model_architecture'])(channels=args['nchannels'],  # Use 4x fewer channels to train faster.
                              num_heads=args['num_heads'], # 8
                              num_transformer_layers=args['num_transformer_layers'], # 11
                              heads_channels = {'target':batch['target'].shape[2]},
                              pooling_type=args['pooling_type'], # max
                              out_activation=args['out_activation'],
                              side_trunk_depth=args['side_trunk_depth'],
                             )

    global_step = tf.Variable(1.)
    ckpt_info = {'global_step': tf.Variable(1.), 'latest_checkpoint':''}
    my_ckpt = tf.train.Checkpoint(step=ckpt_info['global_step'],net=model,iterator=data_it
                                 )
    manager = tf.train.CheckpointManager(my_ckpt, checkpoint, max_to_keep=3)
    if manager.latest_checkpoint:
#         print('Restoring checkpoint at: '+manager.latest_checkpoint)
        my_ckpt.restore(manager.latest_checkpoint).expect_partial()
    else:
        print('Did not find a stored checkpoint!')
    ckpt_info['latest_checkpoint']=manager.latest_checkpoint

    return model,ckpt_info,datasets,args

# def plot_checkpoint_browser(checkpoint,nvalid=1):
#     model,ckpt_info,datasets,args = restore_checkpoint(checkpoint)
    
#     # Plot browser tracks for an example
#     data_it_valid=iter(datasets['valid'])
#     for i in range(nvalid):
#         batch_mc_valid=next(data_it_valid)
#     outputs = model(batch_mc_valid['predictors'], is_training=False)['target']
    
#     # Plot tracks
#     ax=plot_mc(batch_mc_valid['target'].numpy().squeeze(),col=['k','k'],figsize=(20,12))
#     ax=plot_mc(outputs.numpy().squeeze(),ax=ax,col=['r','g'])
#     ax.legend(np.array(ax.lines)[[1,-1]],['mCG','mCH'])
#     global_step=ckpt_info["global_step"].numpy()
#     ax.set_title(': '.join(ckpt_info['latest_checkpoint'].split('/')[-2:]) + f'\nTrained for {global_step} steps')
    
#     return ax,ckpt_info

# def plot_checkpoint_correlations(checkpoint,nvalid=64):
#     model,ckpt_info,datasets,args = restore_checkpoint(checkpoint)
    
#     targets,predictions=get_predictions(model,datasets,nvalid=nvalid)
#     fig,axs,cc=plot_cc(targets,predictions)
#     global_step=ckpt_info["global_step"].numpy()
#     fig.suptitle(': '.join(ckpt_info['latest_checkpoint'].split('/')[-2:]) + f'\nTrained for {global_step} steps')

#     return fig,axs,ckpt_info,cc

def plot_rna(target_df,
            binsize=128,
            ax=None,
            figsize=(12,12),
            cellorder=None,
             yspace=1,
             cmap=['k'],
             linewidth=.5
           ):
    # Browser plot of tracks
#     target = target[:,cell_order]
    posvec = np.arange(target_df.shape[0])*binsize
    classes = target_df.columns.unique()
    target_log = np.log10(target+1)
    target_log = target_log-target_log.median(axis=0)
#     target_log = target
#     target_log[target<=np.percentile(target.values,20)]=np.nan
    
#     if cellorder is None:
#         cellorder=np.arange(target.shape[0])
    if ax is None:
        plt.figure(figsize=figsize)
        ax=plt.gca()

    from itertools import cycle
    if cmap is None:
        cmap=plt.get_cmap("tab10").colors
    cmap = cycle(cmap)
    cmap = {c: next(cmap) for c in classes}
    for i in range(target_log.shape[1]):
        ax.plot(posvec, target_log.iloc[:,i]/yspace + i,color=cmap[target.columns[i]],linewidth=linewidth)
    ax.set_xlim([posvec[0],posvec[-1]])
    ax.set_xlabel('Position (bp)')
    ax.set_yticks(np.arange(target_df.shape[1])+.5)
    ax.set_yticklabels(target_log.iloc[cell_order].columns)
    return ax

# def plot_mc(target_df,
#             mc_track_metadata=mc_track_metadata,
#             binsize=128,
#             ax=None,
#             col=['b','g'],
#             figsize=(12,12)
#            ):
#     # Browser plot of tracks
#     for ctxt in ['cg','ch']:
#         target_df
#     mcg = mc[:,::2]
#     mch = mc[:,1::2]
#     mcg = mcg[:,cell_order]
#     mch = mch[:,cell_order]
#     posvec = np.arange(mc.shape[0])*binsize
#     yspace = [1,.1]
#     xspace=posvec[-1]+binsize*5
    
# #     fig,axs=plt.subplots(2,1,sharex=True,figsize=(8,12))
#     if ax is None:
#         plt.figure(figsize=figsize)
#         ax=plt.gca()
#     ax.plot(posvec, mcg/yspace[0] + np.arange(mcg.shape[1]),col[0],label='mCG');
#     ax.plot(posvec+xspace, mch/yspace[1] + np.arange(mch.shape[1]),col[1],label='mCH');
#     ax.set_xlim([posvec[0],posvec[-1]+xspace])
#     ax.set_xlabel('Position (bp)')
#     ax.set_yticks(np.arange(mcg.shape[1])+.5)
#     ax.set_yticklabels(mc_track_metadata['celltype'].iloc[::2].iloc[cell_order])
#     return ax

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# def plot_cc(target, prediction):
#     cells_show=np.array(cell_order)
#     ncells=cells_show.max()
#     cells_show=np.concatenate((cells_show,cells_show+ncells+1))

#     cc = np.corrcoef(target.numpy().squeeze().T, 
#                      prediction.numpy().squeeze().T)
#     ntracks = prediction.shape[-1]
#     ncelltypes=ntracks//2
#     cc_mcg=cc[::2,::2][cells_show,:][:,cells_show]
#     cc_mch=cc[1::2,1::2][cells_show,:][:,cells_show]

#     fig,axs_rows=plt.subplots(3,3,figsize=(15,20),sharex=False,sharey=False,gridspec_kw={'width_ratios': [1,1,.1]})
#     axs=axs_rows[0]
#     sns.heatmap(cc_mcg,ax=axs[0],xticklabels=False,yticklabels=False,square=True,vmin=0,vmax=1,cbar=False)
#     sns.heatmap(cc_mch,ax=axs[1],xticklabels=False,yticklabels=False,square=True,vmin=0,vmax=1,
#                 cbar=True,cbar_ax=axs[2],cbar_kws={'label':'Correlation','ticks':[0,.5,1],'shrink':0.2})
# #     axs[1].sharex(axs[0])
# #     axs[1].sharey(axs[0])
#     for ax in axs[:2]:
#         ax.vlines(ncells+1,0,2*(ncells+1),'w')
#         ax.hlines(ncells+1,0,2*(ncells+1),'w')
#         ax.set_xticks([ntracks/4,3*ntracks/4])
#         ax.set_xticklabels(['Data','Prediction'])
#     axs[0].set_yticks([ntracks/4,3*ntracks/4])
#     axs[0].set_yticklabels(['Data','Prediction'])
#     axs[0].set_title('mCG')
#     axs[1].set_title('mCH')

#     axs=axs_rows[1]
#     cell_labels=mc_track_metadata['celltype'].iloc[::2].iloc[cell_order]
#     for ax,ccu in zip(axs[:2],[cc_mcg,cc_mch]):
#         ccuu=ccu[:ncelltypes,ncelltypes:]
#         cax=inset_axes(ax,width="5%",height="25%",loc='upper left',borderpad=0,bbox_to_anchor=(1.05, 0., 1, 1),bbox_transform=ax.transAxes)
#         sns.heatmap(ccuu,ax=ax,
#                     xticklabels=cell_labels,yticklabels=cell_labels,
#                     square=True,
#                     vmin=np.percentile(ccuu,5),vmax=np.percentile(ccuu,95),
#                     cbar=True,cbar_ax=cax
#                    )
#         ax.set_xticklabels(ax.get_xticklabels(),fontsize=7)
#         ax.set_yticklabels(ax.get_yticklabels(),fontsize=7)
#         ax.set_title('Data vs. Prediction')
#     axs[0].set_title('mCG')
#     axs[1].set_title('mCH')
#     axs[2].remove()

#     axs=axs_rows[2]
# #     print(cc_mcg[:ncelltypes,:ncelltypes].squeeze(),cc_mcg[:ncelltypes,ncelltypes:].squeeze())
#     for ax,ccu in zip(axs[:2], [cc_mcg,cc_mch]):
#         hh1=ax.plot(ccu[:ncelltypes,:ncelltypes].squeeze(),ccu[:ncelltypes,ncelltypes:].squeeze(),'.',color='gray')
#         hh2=ax.plot(np.diag(ccu[:ncelltypes,:ncelltypes]),np.diag(ccu[:ncelltypes,ncelltypes:]),'ro')
#         ax.grid(True)
#         ax.sharex(axs[0])
#         ax.sharey(axs[0])
#         ax.set_xlabel('Correlation (data x data)')
        
#         rmean=np.diag(ccu[:ncelltypes,ncelltypes:]).mean()
#         rstd=np.diag(ccu[:ncelltypes,ncelltypes:]).std()
#         ax.legend([hh1[0],hh2[0]],['Cell type vs. other cell types',f'Cell type vs. itself (r={rmean:.2}±{rstd:.2})'])
#         ax.plot([0,1],[0,1],'k-')
#         ax.set_xlim([-0.01,1.01])
#         ax.set_ylim([-0.01,1.01])
#     axs[0].set_ylabel('Correlation (data x prediction)')
#     axs[2].remove()
    
#     return fig,axs,cc

# Plotting routines
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_checkpoint_browser(checkpoint,nvalid=1,cell_order=None,figsize=(12,12)):
    df=get_predictions_df(ckpt,nvalid=nvalid)
#     targets_metadata=get_targets_metadata(args['targets_dir'])

    # Plot tracks
    ax=plot_rna(df['targets'],cmap=['gray'],linewidth=1,figsize=figsize)
    ax=plot_rna(df['predictions'],cmap=plt.get_cmap('tab10').colors,ax=ax,linewidth=1)
#     ax=plot_rna(df['targets'],cmap=['gray'],linewidth=1,figsize=figsize)
#     ax=plot_rna(df['predictions'],cmap=plt.get_cmap('tab10').colors,ax=ax,linewidth=1)
    global_step=ckpt_info["global_step"].numpy()
    ax.set_title(checkpoint.split('/')[-1] + f'\nTrained for {global_step} steps')
    
    return ax,ckpt_info

def plot_checkpoint_correlations(checkpoint,nvalid=64):
    model,ckpt_info,datasets,args = restore_checkpoint(checkpoint)
    
    targets,predictions=get_predictions(model,datasets,nvalid=nvalid)
    fig,axs,cc=plot_cc(targets,predictions)
    global_step=ckpt_info["global_step"].numpy()
    fig.suptitle(': '.join(ckpt_info['latest_checkpoint'].split('/')[-2:]) + f'\nTrained for {global_step} steps')

    return fig,axs,ckpt_info,cc

def plot_cc(target, prediction):
    cells_show=np.array(cell_order)
    ncells=cells_show.max()
    cells_show=np.concatenate((cells_show,cells_show+ncells+1))

    cc = np.corrcoef(target.numpy().squeeze().T, 
                     prediction.numpy().squeeze().T)
    ntracks = prediction.shape[-1]
    ncelltypes=ntracks//2
    cc_mcg=cc[::2,::2][cells_show,:][:,cells_show]
    cc_mch=cc[1::2,1::2][cells_show,:][:,cells_show]

    fig,axs_rows=plt.subplots(3,3,figsize=(15,20),sharex=False,sharey=False,gridspec_kw={'width_ratios': [1,1,.1]})
    axs=axs_rows[0]
    sns.heatmap(cc_mcg,ax=axs[0],xticklabels=False,yticklabels=False,square=True,vmin=0,vmax=1,cbar=False)
    sns.heatmap(cc_mch,ax=axs[1],xticklabels=False,yticklabels=False,square=True,vmin=0,vmax=1,
                cbar=True,cbar_ax=axs[2],cbar_kws={'label':'Correlation','ticks':[0,.5,1],'shrink':0.2})
#     axs[1].sharex(axs[0])
#     axs[1].sharey(axs[0])
    for ax in axs[:2]:
        ax.vlines(ncells+1,0,2*(ncells+1),'w')
        ax.hlines(ncells+1,0,2*(ncells+1),'w')
        ax.set_xticks([ntracks/4,3*ntracks/4])
        ax.set_xticklabels(['Data','Prediction'])
    axs[0].set_yticks([ntracks/4,3*ntracks/4])
    axs[0].set_yticklabels(['Data','Prediction'])
    axs[0].set_title('mCG')
    axs[1].set_title('mCH')

    axs=axs_rows[1]
    cell_labels=targets_metadata['celltype'].iloc[::2].iloc[cell_order]
    for ax,ccu in zip(axs[:2],[cc_mcg,cc_mch]):
        ccuu=ccu[:ncelltypes,ncelltypes:]
        cax=inset_axes(ax,width="5%",height="25%",loc='upper left',borderpad=0,bbox_to_anchor=(1.05, 0., 1, 1),bbox_transform=ax.transAxes)
        sns.heatmap(ccuu,ax=ax,
                    xticklabels=cell_labels,yticklabels=cell_labels,
                    square=True,
                    vmin=np.percentile(ccuu,5),vmax=np.percentile(ccuu,95),
                    cbar=True,cbar_ax=cax
                   )
        ax.set_xticklabels(ax.get_xticklabels(),fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(),fontsize=7)
        ax.set_title('Data vs. Prediction')
    axs[0].set_title('mCG')
    axs[1].set_title('mCH')
    axs[2].remove()

    axs=axs_rows[2]
#     print(cc_mcg[:ncelltypes,:ncelltypes].squeeze(),cc_mcg[:ncelltypes,ncelltypes:].squeeze())
    for ax,ccu in zip(axs[:2], [cc_mcg,cc_mch]):
        hh1=ax.plot(ccu[:ncelltypes,:ncelltypes].squeeze(),ccu[:ncelltypes,ncelltypes:].squeeze(),'.',color='gray')
        hh2=ax.plot(np.diag(ccu[:ncelltypes,:ncelltypes]),np.diag(ccu[:ncelltypes,ncelltypes:]),'ro')
        ax.grid(True)
        ax.sharex(axs[0])
        ax.sharey(axs[0])
        ax.set_xlabel('Correlation (data x data)')
        
        rmean=np.diag(ccu[:ncelltypes,ncelltypes:]).mean()
        rstd=np.diag(ccu[:ncelltypes,ncelltypes:]).std()
        ax.legend([hh1[0],hh2[0]],['Cell type vs. other cell types',f'Cell type vs. itself (r={rmean:.2}±{rstd:.2})'])
        ax.plot([0,1],[0,1],'k-')
        ax.set_xlim([-0.01,1.01])
        ax.set_ylim([-0.01,1.01])
    axs[0].set_ylabel('Correlation (data x prediction)')
    axs[2].remove()
    
    return fig,axs,cc

def get_predictions_df(ckpt,nvalid=32,column_names='cellclass'):
    model,ckpt_info,datasets,args = restore_checkpoint(ckpt)
    
    # Plot browser tracks for an example
    df={}
    df['targets'],df['predictions']=get_predictions(model,datasets,nvalid=nvalid)
    targets_metadata=get_targets_metadata(args['targets_dir'])
    for k in ['targets','predictions']:
        df[k]=pd.DataFrame(df[k].numpy(),columns=targets_metadata[column_names])
        df[k]=df[k].sort_index(axis=1)
    return df

def get_targets_metadata(targets_dir):
    targets_metadata = pd.read_csv(targets_dir+'/targets.txt',sep='\t',index_col=0)
    targets_metadata['celltype']=targets_metadata.description.str.extract('C[0-9]+\.[0-9]_(.*)$')

    # Add metadata about cell class
    if '_mc_' in targets_dir:
        targets_metadata['celltype']=targets_metadata.description.str.extract('(.*)_c[gh]')
        targets_metadata['mc_type']=targets_metadata.description.str.extract('(c[gh])')
    targets_metadata['cellclass']=targets_metadata['celltype']
    targets_metadata['cellclass']=targets_metadata['cellclass'].str.replace('L2_3','L2/3').str.replace('_IT','-IT').str.replace('_NP','-NP')
    targets_metadata['cellclass']=targets_metadata['cellclass'].str.replace('_CT','-CT').str.replace('_ET','-ET')
    targets_metadata['cellclass']=targets_metadata['cellclass'].str.replace(r'_.*$','',regex=True)
    

    return targets_metadata
