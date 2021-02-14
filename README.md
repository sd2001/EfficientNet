# EfficientNet-Rescon
Implementation of EfficientNet: Rethinking Model Scaling for CNNs 

Developed by Team Cygnus in SRM MIC Rescon 1.0

<div>  
  <img align="right" width="150" height="150" src="https://github.com/sd2001/EfficientNet-Rescon/blob/main/imgs/bs.png">
 </div> 
<strong>Team Members:</strong>

- <a href="https://github.com/PrathameshDeshpande">Prathamesh Deshpande</a>
- <a href="https://github.com/sd2001">Swarnabha Das</a>
- <a href="https://github.com/norserambler">Yudhajeet Bhattacharya</a>


Click »<a href="https://arxiv.org/pdf/1905.11946v5.pdf"><strong>here</strong></a>« for the original paper

Click »<a href="https://docs.google.com/presentation/d/1orDLEPOTLfVI5EltMJxhI4yiSQFK488SRkb6umpKSls/edit?usp=sharing"><strong>here</strong></a>« for the presentation

<h2>What is EfficientNet?</h2>
Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet.
EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. 
In particular, the Efficient Net-B6-Wide achieves state-of-the-art 91.12% top-1 accuracy on ImageNet(480M Parameters), while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet.
  
<h2>Inference</h2>
The empirical research on EfficientNets shows that it is critical to balance all dimensions of network width/ depth/ resolution, and surprisingly such balance can be achieved by simply scaling each of them with constant ratio.

<div align="center">
<b>How does Compound Scaling work and is different from old paradigms.<br>
  
![image](https://github.com/sd2001/EfficientNet-Rescon/blob/main/imgs/compound.png)  
<br><br>
- Scaling up any dimension of network width, depth, or resolution improves accuracy, but the accuracy gain diminishes for bigger models.
- In order to pursue better accuracy and efficiency, it is critical to balance all dimensions of network width, depth, and resolution during ConvNet scaling.
Intuitively, the compound scaling method makes sense because if the input image is bigger, then the network needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on the bigger image


<b>How Efficient net performs when compared to past transfer learning models.<br>Currently are by far the EfficientNets 

![image](https://github.com/sd2001/EfficientNet-Rescon/blob/main/imgs/comparison.png)
</div>
Earlier GPipes were provided State of Art Results on ImageNet, but are computationally the most expensive. Currently EfficientNets are by far the best with Efficientnet-B6 wide and EfficientNet L2 ticking top accuracies of 91.12% and 91.02% respectively

<h2>Acknowledgements</h2>

- https://github.com/qubvel/efficientnet

- https://medium.com/@nainaakash012/efficientnet-rethinking-model-scaling-for-convolutional-neural-networks-92941c5bfb95

- https://medium.com/analytics-vidhya/image-classification-with-efficientnet-better-performance-with-computational-efficiency-f480fdb00ac6

- https://www.youtube.com/watch?v=IBndcd4UfTs

- https://www.youtube.com/watch?v=4nqcufewUlk&t=3098s

<h2>To run it locally</h2>

- Fork the project

- Open Terminal in your desired folder

```
git clone https://github.com/sd2001/EfficientNet.git

cd EfficientNet

python3 -m venv env

source env/bin/activate

pip install -r requirements.txt

cd src
 
python architecture.py
```

**To check the Notebooks**

In the Efficient Folder

```
jupyter notebook
```

