#%%
#get the predictions for the test data
predicted_classes = model.predict_classes(X_test)

#get the indices to be plotted
y_true = data_test.iloc[:, 0]

from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]

cfr = classification_report(y_true, predicted_classes, target_names=target_names)

df_cfr = classification_report_dataframe(cfr)


class_labels = list(label_nums.values())
#df_cfr['class'] = class_labels
df_cfr.index = class_labels
df_cfr.drop('class',axis=1,inplace=True)

# Write to Excel
path_classification_rep = os.path.join(path_run,run_name + "classification report"+".xlsx")
with pd.ExcelWriter(path_classification_rep) as writer:
    df_cfr.to_excel(writer,'classification')
    writer.save()

#%% Heatmap
fig, ax = plt.subplots(figsize=LANDSCAPE_A4);         # Sample figsize in inches

sns.heatmap(df_cfr.iloc[:,0:3], annot=True, fmt="g", cmap='inferno',ax=ax,square = True)
ax.xaxis.set_ticks_position('top')
fig.suptitle("Classification report, "+run_name)
plt.show()

# Save figure   
path_this_report_out = os.path.join(path_run,run_name + ' classification'+".pdf")
with open(path_this_report_out, 'wb') as f:
    plt.savefig(f,dpi=600,format = 'pdf')
logging.debug("Wrote to {}".format(path_this_report_out))


#%%
correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]

#%% Correctly predicted classes.

plt.style.use('ggplot')


plot_rows = 4
plot_cols = 5
f, axs = plt.subplots(plot_rows, plot_cols, figsize=LANDSCAPE_A3,sharey=False,facecolor='0.15')

cnt = 0
for i, this_row in enumerate(axs):
    for j, ax in enumerate(this_row):
        this_img_index =  correct[cnt]
        #print(this_img_index)
        this_img = X_test[this_img_index].reshape(28,28)
        print("Row, Col;", i,j, "Image index:", this_img_index)
        #print(ax)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(this_img,cmap='gray')
        ax_title_str = "{} {}".format(this_img_index,label_nums[y_true[this_img_index]])
        ax.set_title(ax_title_str)
        cnt+=1
f.suptitle(run_name + " correct predictions",fontsize=16)
plt.show()

path_this_report_out = os.path.join(path_run,run_name + " correct predictions"+".pdf")
with open(path_this_report_out, 'wb') as f:
    plt.savefig(f,dpi=600,format = 'pdf')
logging.debug("Wrote to {}".format(path_this_report_out))

    #ax.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')

#%% INCORRECT predicted classes.

plt.style.use('ggplot')

plot_rows = 4
plot_cols = 5
f, axs = plt.subplots(plot_rows, plot_cols, figsize=LANDSCAPE_A3,sharey=False,facecolor='0.15')

cnt = 0
for i, this_row in enumerate(axs):
    for j, ax in enumerate(this_row):
        this_img_index =  incorrect[cnt]
        #print(this_img_index)
        this_img = X_test[this_img_index].reshape(28,28)
        print("Row, Col;", i,j, "Image index:", this_img_index)
        #print(ax)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(this_img,cmap='gray')
        ax_title_str = "{} {}".format(this_img_index,label_nums[y_true[this_img_index]])
        ax.set_title(ax_title_str)
        cnt+=1
f.suptitle(run_name + " incorrect predictions",fontsize=16)
plt.show()

#
path_this_report_out = os.path.join(path_run,run_name + " incorrect predictions"+".pdf")
with open(path_this_report_out, 'wb') as f:
    plt.savefig(f,dpi=600,format = 'pdf')
logging.debug("Wrote to {}".format(path_this_report_out))

#%%

#classification_report, , f1_score

def plot_confusion_matrix(cm,
                          class_,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function plots a confusion matrix
    """
    f, ax = plt.subplots(figsize=LANDSCAPE_A4,sharey=False,facecolor='0.15')

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.set_title(title)
    ax.grid(False)
    tick_marks = np.arange(len(class_))
    plt.xticks(tick_marks, class_, rotation=90)
    plt.yticks(tick_marks, class_)
    f.suptitle(title,fontsize=16)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.show()
class_labels = list(label_nums.values())
conf_matrix = sk.metrics.confusion_matrix(y_true, predicted_classes)
plot_confusion_matrix(cm=conf_matrix,
                      class_=class_labels,
                      title=run_name + ' confusion matrix',
                      cmap='inferno_r')
#
path_this_report_out = os.path.join(path_run,run_name + " confusion matrix"+".pdf")
with open(path_this_report_out, 'wb') as f:
    plt.savefig(f,dpi=600,format = 'pdf')
logging.debug("Wrote to {}".format(path_this_report_out))
