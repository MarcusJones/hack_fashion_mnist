# -*- coding: utf-8 -*-



#%%
small = r"/home/batman/git/hack_fashion_mnist/Runs/smaller_actual"
model_path = os.path.join(small,"model_weights.h5")
model_arch = os.path.join(small,"model_architecture.json")

json_file = open(model_arch, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = ks.models.model_from_json(loaded_model_json)
model.summary()

vis_path = os.path.join(small,'small visualized.png')
ks.utils.vis_utils.plot_model(model, to_file=vis_path, show_shapes=True, show_layer_names=False)

#%%
baseline = r"/home/batman/git/hack_fashion_mnist/Runs/baseline"
model_path = os.path.join(baseline,"model_weights.h5")
model_arch = os.path.join(baseline,"model_architecture.json")

#model_path = os.path.join(baseline,"model_weights.h5")
#model = ks.models.load_model(model_path)
#model.summary()
json_file = open(model_arch, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = ks.models.model_from_json(loaded_model_json)
model.summary()

vis_path = os.path.join(baseline,'baseline visualized.png')
ks.utils.vis_utils.plot_model(model, to_file=vis_path, show_shapes=True, show_layer_names=False)

#%%
bigger = r"/home/batman/git/hack_fashion_mnist/Runs/bigger"
model_path = os.path.join(bigger,"model_weights.h5")
model_arch = os.path.join(bigger,"model_architecture.json")

#model_path = os.path.join(baseline,"model_weights.h5")
#model = ks.models.load_model(model_path)
#model.summary()
json_file = open(model_arch, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = ks.models.model_from_json(loaded_model_json)
model.summary()

vis_path = os.path.join(bigger,'bigger visualized.png')
ks.utils.vis_utils.plot_model(model, to_file=vis_path, show_shapes=True, show_layer_names=False)
