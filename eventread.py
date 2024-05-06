from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json
import os


output_path = './sgd_result_23'
target_folder = output_path
if not os.path.isdir(target_folder):
    os.makedirs(target_folder)
dataset = str('mnist')
istanh = False
if istanh:
    result_json_file = dataset  + '_realexp_tanh2' + '.json'
    result_json_file = os.path.join(target_folder, result_json_file)
else:
    # result_json_file = 'phi_'+ args.phi +'p_' + str(p) + "_n_" + str(n) +'.json'
    result_json_file = dataset  + '_sgd_second' + '.json'
    result_json_file = os.path.join(target_folder, result_json_file)
    # Read existing data from the JSON file, or initialize as an empty list
try:
    with open(result_json_file, 'r') as json_file:
        existing_data = json.load(json_file)
except (FileNotFoundError, json.decoder.JSONDecodeError):
    existing_data = []

# 定义要读取的TensorBoard文件路径
logdir = './result_sgd_second/'

dim_list = [32,128,256,1024,4096]
for dim in dim_list:
    model_list = [str('tanh_inn'), str('tanh_enn'), str('h_tanh_enn'), str('inn'), str('l_relu_enn'), str('relu_enn')]
    # model = str('tanh_inn')
    for model in model_list:
        lr = str(1.0)
        # 用来搜索需要的模型，并保存
        targer_search = str(dim) + model + lr + dataset
        # 遍历文件夹中的所有文件
        for filename in os.listdir(logdir):
            if targer_search in filename:
                file_path = os.path.join(logdir, filename)
                # 创建EventAccumulator对象并加载TensorBoard文件
                event_acc = EventAccumulator(file_path)
                event_acc.Reload()

                # 获取TensorBoard文件中的所有标签（即事件类型）
                tags = event_acc.Tags()

                # 选择要读取的标签
                tag_list = ['Accuracy/Test']  # 示例：读取标量数据
                for tag in tag_list:
                    # 从TensorBoard文件中读取指定标签的数据
                    data = []
                    # target_steps = [9, 99]
                    for event in event_acc.Scalars(tag):
                        # if int(event.step) in target_steps:
                        # if int(event.step) % 2 == 0:
                        data.append({
                            'step': int(event.step),
                            'value': float(event.value),
                            'tag':str(tag),
                            'dim':str(dim),
                            'model':model,
                            'lr':lr,
                            'dataset':dataset
                        })
                    existing_data.append(data)
                with open(result_json_file, 'w') as json_file:
                    json.dump(existing_data, json_file, indent=4)

                