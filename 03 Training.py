history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, y_val))


#%% Eval on TEST
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

path_text_summary_out = os.path.join(path_run,run_name+" summary"+".txt")
with open(path_text_summary_out, 'a') as f:
    print('Test loss:', score[0],file=f)
    print('Test accuracy:', score[1],file=f)
#%% Save history
total_hist = history.__dict__.copy()
total_hist.pop('validation_data', None)
hist_model = total_hist.pop('model', None)

json_path = os.path.join(path_run,r"model_history.json")

    #with open(path_history, 'w') as fp:
    #    json_string = json.dump(history_dict,fp)
        
with open(json_path, "w") as json_file:
    json.dump(total_hist,json_file)
logging.info("Saved history dictionary to {}".format(json_path))


#%% Save model weights
model_path = os.path.join(path_run,r"model_weights.h5")
model.save(model_path)  # creates a HDF5 file 'my_model.h5'

#del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
#model = ks.models.load_model(model_path)
