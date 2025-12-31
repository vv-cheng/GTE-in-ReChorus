# GTE-in-ReChorus
机器学习大作业

林泽熙 23303041

范程炜 23330025

# 实验环境
配置如requirements.txt所示,与ReChorus的配置环境基本一致，唯一不同的是使用了cudatoolkit==11.3.1.

环境配置使用了如下代码：

conda create -n rechorus python==3.10.4 pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 numpy==1.22.3 ipython==8.10.0 jupyter==1.0.0 pandas==1.4.4 scikit-learn==1.1.3 scipy==1.7.3 pyyaml -c pytorch

pip install tqdm==4.66.1

# 数据集
我们将论文中的数据集进行了转换使得能在ReChorus框架中使用,转换代码为.\data\dataGTE\convert.py.

同时.\data\dataGTE包含了论文中的原始数据集为pkl格式,转换后的数据集将位于.\data.

若需要使用转换代码,转换代码的运行方式如下：

cd .\data\dataGTE

python convert.py --all

注意：实验环境由于scipy==1.7.3无法转换tmall数据集，请使用较新版本的scipy.

# 模型

我们在ReChorus框架中实现了论文中的Graph Topology Encoder(GTE)算法,以及用于对比的部分算法GCCF和SimGRACE.

GTE算法代码包含了.\src\model\general\GTE.py和.\src\helpers\GTERunner.py

GCCF算法代码为.\src\model\general\GCCF.py

SimGRACE算法代码为.\src\model\general\SimGRACE.py

模型运行方式,与ReChorus框架一致,包含相同的公共命令行参数,如下：

python .\src\main.py --model_name 模型名 --dataset 数据集名


# 批量运行

若要进行批量运行，可以使用exp.py,运行代码为：

python ./src/exp.py --in_f run.sh --out_f result.csv --gpu 0

更多命令行参数详见exp.py

同时我们也上传了我们的批量运行文件run.sh
