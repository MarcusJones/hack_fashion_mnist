#%%

def plot_hist_dict(history_dict, model_title = ''):
    
    assert 'epoch' in history_dict
    assert 'history' in history_dict
    
    #model_title = "10 Epochs"
    #fig = plt.figure(figsize=(5,4))
    #fig=plt.figure(figsize=(20, 10),facecolor='white')
    
    

    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5),sharey=False,facecolor='white')
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=LANDSCAPE_A3,sharey=False,facecolor='0.15')
    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=PORTRAIT_A3,sharey=False,facecolor='0.15')
    
    ax1.plot(history_dict['epoch'],  history_dict['history']['loss'],label="Train")
    ax1.plot(history_dict['epoch'],  history_dict['history']['val_loss'],label="CV")
    ax1.set_title("Loss function development - Training set vs CV set")
    ax1.legend(loc='upper right')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Values')
    ax1.set_ylim([0,1])
    
    ax2.plot(history_dict['epoch'],  history_dict['history']['acc'],label="Train")
    ax2.plot(history_dict['epoch'],  history_dict['history']['val_acc'],label="CV")
    ax2.set_title("Accuracy development - Training set vs CV set")
    ax2.legend(loc='upper right')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Values')
    ax2.set_ylim([0.5,1])
    f.suptitle(model_title, fontsize=16)
    #plt.tight_layout(pad=5)
    #this_plot = plt.suptitle(model_title, fontsize=16)
    
    return f 
    #plt.show()

#%%
title_str = "Train vs CV development, " + run_name
f = plot_hist_dict(history.__dict__,title_str)
f.show()

#%% Save figure   
path_this_report_out = os.path.join(path_run,title_str+".pdf")
with open(path_this_report_out, 'wb') as f:
    plt.savefig(f,dpi=600,format = 'pdf')
logging.debug("Wrote to {}".format(path_this_report_out))
