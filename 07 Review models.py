# -*- coding: utf-8 -*-



#%%
small = r"smaller_actual"
model_arch = os.path.join(PROJECT_ROOT,'Runs',small,"model_architecture.json")


json_file = open(model_arch, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = ks.models.model_from_json(loaded_model_json)
model.summary()

vis_path = os.path.join(small,'small visualized.png')
ks.utils.vis_utils.plot_model(model, to_file=vis_path, show_shapes=True, show_layer_names=False)


vis_path = os.path.join(PROJECT_ROOT,'Runs',small,'small visualized HOR.png')
ks.utils.vis_utils.plot_model(model, to_file=vis_path, show_shapes=True, show_layer_names=False,rankdir='LR')


#%%
baseline = r"baseline"
model_arch = os.path.join(PROJECT_ROOT,'Runs',baseline,"model_architecture.json")

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

vis_path = os.path.join(PROJECT_ROOT,'Runs',baseline,'baseline visualized HOR.png')
ks.utils.vis_utils.plot_model(model, to_file=vis_path, show_shapes=True, show_layer_names=False,rankdir='LR')

#%%
bigger = r"bigger100"
model_arch = os.path.join(PROJECT_ROOT,'Runs',bigger,"model_architecture.json")

#model_path = os.path.join(baseline,"model_weights.h5")
#model = ks.models.load_model(model_path)
#model.summary()
json_file = open(model_arch, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = ks.models.model_from_json(loaded_model_json)
model.summary()

vis_path = os.path.join(PROJECT_ROOT,'Runs',bigger,'bigger visualized VERT.png')
ks.utils.vis_utils.plot_model(model, to_file=vis_path, show_shapes=True, show_layer_names=False)

vis_path = os.path.join(PROJECT_ROOT,'Runs',bigger,'bigger visualized HOR.png')
ks.utils.vis_utils.plot_model(model, to_file=vis_path, show_shapes=True, show_layer_names=False,rankdir='LR')
