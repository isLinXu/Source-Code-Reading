# 实体抽取任务案例
为方便用户快速上手，下面我们准备了一个真实案例来演示如何使用`Hunyuan-Large`进行精调。
- 基座模型：`Hunyuan-Large-Instruct`
- 训练数据：约5k条汽车领域的[实体抽取数据](data)
- 训练数据示例如下
```
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "请提取下列文章中包含的车系：\n【恒升新迪比亚迪精诚服务】秦宋新能源车主爱车讲堂——春季公开课圆满结束\n"}, {"role": "assistant", "content": "秦；宋新能源"}]}
```

## 训练部分
- 训练环境配置可参考 [train/README.md](../train/README.md)
- 训练脚本可参考 [train_demo.sh](train_demo.sh)，需要修改其中的`model_path`、`train_data_file`和`output_path`
- 首次加载模型耗时会较长，成功运行后会打印每个step的loss
```
 {'loss': 7.4291, 'grad_norm': 144.42880249023438, 'learning_rate': 5e-06, 'epoch': 0.03}
 {'loss': 7.4601, 'grad_norm': 141.73260498046875, 'learning_rate': 4.998892393243008e-06, 'epoch': 0.06}
 {'loss': 3.2492, 'grad_norm': 32.29960250854492, 'learning_rate': 4.995570688238146e-06, 'epoch': 0.08}
 {'loss': 2.984, 'grad_norm': 39.98820495605469, 'learning_rate': 4.990038229660787e-06, 'epoch': 0.11}
 {'loss': 1.8134, 'grad_norm': 23.492595672607422, 'learning_rate': 4.98230058822775e-06, 'epoch': 0.14}
 {'loss': 1.5473, 'grad_norm': 23.56238555908203, 'learning_rate': 4.972365555088068e-06, 'epoch': 0.17}
 {'loss': 0.7976, 'grad_norm': 21.16975212097168, 'learning_rate': 4.960243133977955e-06, 'epoch': 0.19}
 {'loss': 0.3414, 'grad_norm': 11.497740745544434, 'learning_rate': 4.945945531147896e-06, 'epoch': 0.22}
 {'loss': 0.2019, 'grad_norm': 3.659959077835083, 'learning_rate': 4.929487143071984e-06, 'epoch': 0.25}
 {'loss': 0.1355, 'grad_norm': 2.6742758750915527, 'learning_rate': 4.910884541951894e-06, 'epoch': 0.28}
 {'loss': 0.1406, 'grad_norm': 1.70888352394104, 'learning_rate': 4.89015645903008e-06, 'epoch': 0.31}
 {'loss': 0.1394, 'grad_norm': 1.9664640426635742, 'learning_rate': 4.8673237657289994e-06, 'epoch': 0.33}
 {'loss': 0.0962, 'grad_norm': 1.2338069677352905, 'learning_rate': 4.84240945263536e-06, 'epoch': 0.36}
 {'loss': 0.0899, 'grad_norm': 1.3094085454940796, 'learning_rate': 4.815438606350553e-06, 'epoch': 0.39}
 {'loss': 0.1535, 'grad_norm': 1.4803948402404785, 'learning_rate': 4.786438384230567e-06, 'epoch': 0.42}
 {'loss': 0.1159, 'grad_norm': 2.006241798400879, 'learning_rate': 4.755437987040832e-06, 'epoch': 0.44}
 {'loss': 0.1434, 'grad_norm': 1.614030122756958, 'learning_rate': 4.722468629553528e-06, 'epoch': 0.47}
 {'loss': 0.1048, 'grad_norm': 1.1502505540847778, 'learning_rate': 4.687563509116949e-06, 'epoch': 0.5}
 {'loss': 0.0783, 'grad_norm': 1.1970840692520142, 'learning_rate': 4.650757772228599e-06, 'epoch': 0.53}
 {'loss': 0.1731, 'grad_norm': 1.5403984785079956, 'learning_rate': 4.612088479145633e-06, 'epoch': 0.56}
 {'loss': 0.1045, 'grad_norm': 0.8574311137199402, 'learning_rate': 4.571594566568329e-06, 'epoch': 0.58}
 {'loss': 0.1122, 'grad_norm': 1.275988221168518, 'learning_rate': 4.529316808434132e-06, 'epoch': 0.61}
 {'loss': 0.0909, 'grad_norm': 0.9738625288009644, 'learning_rate': 4.485297774861752e-06, 'epoch': 0.64}
 {'loss': 0.118, 'grad_norm': 1.3492830991744995, 'learning_rate': 4.439581789286661e-06, 'epoch': 0.67}
 {'loss': 0.1547, 'grad_norm': 1.4021662473678589, 'learning_rate': 4.392214883831154e-06, 'epoch': 0.69}
 {'loss': 0.0944, 'grad_norm': 1.2680057287216187, 'learning_rate': 4.343244752953907e-06, 'epoch': 0.72}
 {'loss': 0.118, 'grad_norm': 1.0716216564178467, 'learning_rate': 4.292720705425691e-06, 'epoch': 0.75}
 {'loss': 0.124, 'grad_norm': 1.1935311555862427, 'learning_rate': 4.240693614679628e-06, 'epoch': 0.78}
 {'loss': 0.1164, 'grad_norm': 1.0407761335372925, 'learning_rate': 4.18721586758595e-06, 'epoch': 0.81}
 {'loss': 0.09, 'grad_norm': 0.8492897748947144, 'learning_rate': 4.132341311702867e-06, 'epoch': 0.83}
 {'loss': 0.0959, 'grad_norm': 0.8767865300178528, 'learning_rate': 4.076125201056637e-06, 'epoch': 0.86}
 {'loss': 0.1053, 'grad_norm': 0.9856031537055969, 'learning_rate': 4.018624140505443e-06, 'epoch': 0.89}
 {'loss': 0.1514, 'grad_norm': 1.407051920890808, 'learning_rate': 3.959896028743106e-06, 'epoch': 0.92}
 {'loss': 0.1083, 'grad_norm': 0.9481344223022461, 'learning_rate': 3.900000000000001e-06, 'epoch': 0.94}
 {'loss': 0.0963, 'grad_norm': 4.065955638885498, 'learning_rate': 3.838996364499903e-06, 'epoch': 0.97}
 {'loss': 0.1385, 'grad_norm': 1.0660120248794556, 'learning_rate': 3.776946547732703e-06, 'epoch': 1.0}
 {'loss': 0.0418, 'grad_norm': 0.46154797077178955, 'learning_rate': 3.713913028604151e-06, 'epoch': 1.03}
 {'loss': 0.0737, 'grad_norm': 0.9685096144676208, 'learning_rate': 3.6499592765248833e-06, 'epoch': 1.06}
 {'loss': 0.0302, 'grad_norm': 0.9554418325424194, 'learning_rate': 3.585149687502118e-06, 'epoch': 1.08}
 {'loss': 0.052, 'grad_norm': 0.605120837688446, 'learning_rate': 3.519549519298328e-06, 'epoch': 1.11}
 {'loss': 0.0331, 'grad_norm': 0.5751758813858032, 'learning_rate': 3.4532248257222053e-06, 'epoch': 1.14}
 {'loss': 0.0473, 'grad_norm': 0.5900629758834839, 'learning_rate': 3.386242390118077e-06, 'epoch': 1.17}
 {'loss': 0.0449, 'grad_norm': 0.6361717581748962, 'learning_rate': 3.3186696581207407e-06, 'epoch': 1.19}
 {'loss': 0.0818, 'grad_norm': 1.0822972059249878, 'learning_rate': 3.25057466974342e-06, 'epoch': 1.22}
 {'loss': 0.0418, 'grad_norm': 0.6318709850311279, 'learning_rate': 3.182025990867247e-06, 'epoch': 1.25}
 {'loss': 0.0417, 'grad_norm': 0.60468590259552, 'learning_rate': 3.113092644201228e-06, 'epoch': 1.28}
 {'loss': 0.0633, 'grad_norm': 0.5810320973396301, 'learning_rate': 3.0438440397822245e-06, 'epoch': 1.31}
 {'loss': 0.026, 'grad_norm': 0.4366144835948944, 'learning_rate': 2.9743499050849347e-06, 'epoch': 1.33}
 {'loss': 0.043, 'grad_norm': 0.8081514239311218, 'learning_rate': 2.9046802148122338e-06, 'epoch': 1.36}
 {'loss': 0.046, 'grad_norm': 0.46932360529899597, 'learning_rate': 2.8349051204365774e-06, 'epoch': 1.39}
 {'loss': 0.067, 'grad_norm': 1.0018138885498047, 'learning_rate': 2.7650948795634223e-06, 'epoch': 1.42}
 {'loss': 0.055, 'grad_norm': 0.8858521580696106, 'learning_rate': 2.6953197851877672e-06, 'epoch': 1.44}
 {'loss': 0.0256, 'grad_norm': 0.5397916436195374, 'learning_rate': 2.6256500949150655e-06, 'epoch': 1.47}
 {'loss': 0.0414, 'grad_norm': 0.5249439477920532, 'learning_rate': 2.556155960217776e-06, 'epoch': 1.5}
 {'loss': 0.0623, 'grad_norm': 1.9616726636886597, 'learning_rate': 2.486907355798773e-06, 'epoch': 1.53}
 {'loss': 0.0529, 'grad_norm': 0.49948468804359436, 'learning_rate': 2.4179740091327534e-06, 'epoch': 1.56}
 {'loss': 0.0395, 'grad_norm': 0.658088207244873, 'learning_rate': 2.3494253302565808e-06, 'epoch': 1.58}
 {'loss': 0.0418, 'grad_norm': 0.5808509588241577, 'learning_rate': 2.28133034187926e-06, 'epoch': 1.61}
 {'loss': 0.0484, 'grad_norm': 0.6675217151641846, 'learning_rate': 2.2137576098819237e-06, 'epoch': 1.64}
 {'loss': 0.0789, 'grad_norm': 1.1550902128219604, 'learning_rate': 2.146775174277796e-06, 'epoch': 1.67}
 {'loss': 0.0246, 'grad_norm': 0.8917452096939087, 'learning_rate': 2.0804504807016725e-06, 'epoch': 1.69}
 {'loss': 0.0351, 'grad_norm': 0.8304968476295471, 'learning_rate': 2.0148503124978823e-06, 'epoch': 1.72}
 {'loss': 0.029, 'grad_norm': 0.6137705445289612, 'learning_rate': 1.950040723475117e-06, 'epoch': 1.75}
 {'loss': 0.0397, 'grad_norm': 0.6633376479148865, 'learning_rate': 1.8860869713958501e-06, 'epoch': 1.78}
 {'loss': 0.0212, 'grad_norm': 0.5623720288276672, 'learning_rate': 1.8230534522672968e-06, 'epoch': 1.81}
 {'loss': 0.0404, 'grad_norm': 0.4446176290512085, 'learning_rate': 1.7610036355000983e-06, 'epoch': 1.83}
 {'loss': 0.0425, 'grad_norm': 0.42774003744125366, 'learning_rate': 1.7000000000000005e-06, 'epoch': 1.86}
 {'loss': 0.0513, 'grad_norm': 1.426970362663269, 'learning_rate': 1.6401039712568944e-06, 'epoch': 1.89}
 {'loss': 0.0376, 'grad_norm': 0.4887981712818146, 'learning_rate': 1.5813758594945576e-06, 'epoch': 1.92}
 {'loss': 0.0312, 'grad_norm': 0.5390689969062805, 'learning_rate': 1.5238747989433645e-06, 'epoch': 1.94}
 {'loss': 0.0421, 'grad_norm': 0.6012405157089233, 'learning_rate': 1.4676586882971339e-06, 'epoch': 1.97}
 {'loss': 0.1072, 'grad_norm': 1.1696946620941162, 'learning_rate': 1.4127841324140512e-06, 'epoch': 2.0}
 {'loss': 0.0352, 'grad_norm': 0.5454437136650085, 'learning_rate': 1.359306385320373e-06, 'epoch': 2.03}
 {'loss': 0.0178, 'grad_norm': 0.3860515356063843, 'learning_rate': 1.3072792945743095e-06, 'epoch': 2.06}
 {'loss': 0.0051, 'grad_norm': 0.133415088057518, 'learning_rate': 1.2567552470460932e-06, 'epoch': 2.08}
 {'loss': 0.0077, 'grad_norm': 0.22402632236480713, 'learning_rate': 1.2077851161688455e-06, 'epoch': 2.11}
 {'loss': 0.0167, 'grad_norm': 0.24073262512683868, 'learning_rate': 1.1604182107133397e-06, 'epoch': 2.14}
 {'loss': 0.0133, 'grad_norm': 0.3704022169113159, 'learning_rate': 1.1147022251382485e-06, 'epoch': 2.17}
 {'loss': 0.0149, 'grad_norm': 0.3235306739807129, 'learning_rate': 1.070683191565868e-06, 'epoch': 2.19}
 {'loss': 0.0202, 'grad_norm': 0.47053492069244385, 'learning_rate': 1.028405433431671e-06, 'epoch': 2.22}
 {'loss': 0.007, 'grad_norm': 0.14351408183574677, 'learning_rate': 9.87911520854368e-07, 'epoch': 2.25}
 {'loss': 0.0171, 'grad_norm': 0.3089752197265625, 'learning_rate': 9.492422277714011e-07, 'epoch': 2.28}
 {'loss': 0.0154, 'grad_norm': 0.2722967565059662, 'learning_rate': 9.124364908830504e-07, 'epoch': 2.31}
 {'loss': 0.0138, 'grad_norm': 0.4130292236804962, 'learning_rate': 8.775313704464731e-07, 'epoch': 2.33}
 {'loss': 0.0065, 'grad_norm': 0.1657753437757492, 'learning_rate': 8.445620129591687e-07, 'epoch': 2.36}
 {'loss': 0.0086, 'grad_norm': 0.1517183482646942, 'learning_rate': 8.135616157694337e-07, 'epoch': 2.39}
 {'loss': 0.0159, 'grad_norm': 0.29184406995773315, 'learning_rate': 7.845613936494468e-07, 'epoch': 2.42}
 {'loss': 0.0034, 'grad_norm': 0.09557823091745377, 'learning_rate': 7.575905473646402e-07, 'epoch': 2.44}
 {'loss': 0.0077, 'grad_norm': 0.2557217478752136, 'learning_rate': 7.326762342710017e-07, 'epoch': 2.47}
 {'loss': 0.0142, 'grad_norm': 0.547609806060791, 'learning_rate': 7.098435409699203e-07, 'epoch': 2.5}
 {'loss': 0.0126, 'grad_norm': 0.3501236140727997, 'learning_rate': 6.89115458048106e-07, 'epoch': 2.53}
 {'loss': 0.0192, 'grad_norm': 0.3290202021598816, 'learning_rate': 6.705128569280162e-07, 'epoch': 2.56}
 {'loss': 0.0184, 'grad_norm': 0.5247418880462646, 'learning_rate': 6.540544688521045e-07, 'epoch': 2.58}
 {'loss': 0.0053, 'grad_norm': 0.14795489609241486, 'learning_rate': 6.397568660220452e-07, 'epoch': 2.61}
 {'loss': 0.0041, 'grad_norm': 0.09380070865154266, 'learning_rate': 6.276344449119325e-07, 'epoch': 2.64}
 {'loss': 0.0343, 'grad_norm': 0.4588293135166168, 'learning_rate': 6.176994117722502e-07, 'epoch': 2.67}
 {'loss': 0.0062, 'grad_norm': 0.1302798092365265, 'learning_rate': 6.099617703392138e-07, 'epoch': 2.69}
 {'loss': 0.0192, 'grad_norm': 0.26435697078704834, 'learning_rate': 6.044293117618545e-07, 'epoch': 2.72}
 {'loss': 0.0105, 'grad_norm': 0.23238405585289001, 'learning_rate': 6.011076067569928e-07, 'epoch': 2.75}
 {'loss': 0.0071, 'grad_norm': 0.1882176399230957, 'learning_rate': 6e-07, 'epoch': 2.78}
 {'train_runtime': 20515.9091, 'train_samples_per_second': 0.624, 'train_steps_per_second': 0.005, 'train_loss': 0.31072488425299527, 'epoch': 2.78}
```

## 部署部分
- 推理环境配置可参考 [推理README.md](../README.md)

## 推理部分
- 推理脚本可参考 [eval_demo_vllm.py](eval_demo_vllm.py)
- 打分脚本可参考 [compute_acc.py](compute_acc.py)