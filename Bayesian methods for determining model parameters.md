# Bayesian methods for determining model parameters

>Reference: 
>- [INFO-6105](./DataScience%20Reference/mcmc-mixture-model-dino-start.ipynb)
>- [打开MCMC（马尔科夫蒙特卡洛）的黑盒子 - Pymc贝叶斯推理底层实现原理初探
](https://www.cnblogs.com/LittleHann/p/9550757.html#_lab2_1_1)
>- [AI pioneer Sejnowski says it’s all about the gradient](https://www.zdnet.com/article/ai-pioneer-sejnowski-says-its-all-about-the-gradient/)

## The Bayesian expectation fabric

When we create a model with  𝑁  parameters that we want to solve with Baeysian inference, we are implicitly creating an  𝑁  dimensional space（**可以理解为N个随机变量**）for the **prior** distribution of each paramater to live in.

Associated with the space is an **extra dimension**, which we can describe as the surface, or manifold, that sits on top of the space, that reflects the probability of observing data. The surface on the space is defined by our prior distribution and warped by the **data likelihood**. I call this fabric the **Bayesian expectation fabric**.

### 先验分布的可视化图景像

我们这里选择2维的，即包含2个随机变量的贝叶斯推断问题，进行可视化展示，选择2维是因为可以方便进行可视化，高维空间是很难想象的。

**二维均匀分布的先验图景像**

For example, if we have two unknown probability distributions  𝑝1  and  𝑝2 , and priors for both are  Uniform(0,5) , the space created is a square of length 5 and the surface is a flat plane that sits on top of the square (representing the concept that every point is equally likely).
```python
%matplotlib inline
import scipy.stats as stats
from IPython.core.pylabtools import figsize
import numpy as np
figsize(12.5, 4)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

jet = plt.cm.jet
fig = plt.figure()
x = y = np.linspace(0, 5, 100)
X, Y = np.meshgrid(x, y)

plt.subplot(121)
uni_x = stats.uniform.pdf(x, loc=0, scale=5) #均匀分布
uni_y = stats.uniform.pdf(y, loc=0, scale=5) #均匀分布
M = np.dot(uni_x[:, None], uni_y[None, :])
im = plt.imshow(M, interpolation='none', origin='lower',
                cmap=jet, vmax=1, vmin=-.15, extent=(0, 5, 0, 5))

plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title("Landscape formed by Uniform priors.")

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X, Y, M, cmap=plt.cm.jet, vmax=1, vmin=-.15)
ax.view_init(azim=390)
plt.title("Uniform prior landscape; alternate view");
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200311222736.png)

**二维指数分布的先验图景像**

Alternatively, if the two priors are exponentials, for example  Exp(3)  and  Exp(10) , then the space is all positive numbers on the 2-D plane, and the surface induced by the priors looks like a **water-fall** that starts at the point (0,0) and flows over the positive numbers. Do you want to see what this looks like? I do.

```python
%matplotlib inline
import scipy.stats as stats
from IPython.core.pylabtools import figsize
import numpy as np
figsize(12.5, 5)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

jet = plt.cm.jet
fig = plt.figure()
x = y = np.linspace(0, 5, 100)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
plt.subplot(121)

exp_x = stats.expon.pdf(x, scale=3)
exp_y = stats.expon.pdf(x, scale=10)
M = np.dot(exp_x[:, None], exp_y[None, :])
CS = plt.contour(X, Y, M)
im = plt.imshow(M, interpolation='none', origin='lower',
                cmap=jet, extent=(0, 5, 0, 5))
#plt.xlabel("prior on $p_1$")
#plt.ylabel("prior on $p_2$")
plt.title("$Exp(3), Exp(10)$ prior landscape")

ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(X, Y, M, cmap=jet)
ax.view_init(azim=390)
plt.title("$Exp(3), Exp(10)$ prior landscape; \nalternate view");
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200311223326.png)

The plots below visualize this. The more dark-red the color, the more prior probability is assigned to that location. Conversely, areas with darker blue represent that our priors assign very low probability to that location.（越红，先验概率越高；越蓝，先验概率越低）

### 观测样本是如何影响未知变量的先验分布的？

概率面描述了未知变量的先验分布，而观测样本的作用我们可以形象地理解为一只手，每次来一个样本，这只手就根据观测样本的情况，将先验分布的曲面向“符合”观测样本的方向拉伸一点。

在MCMC过程中，观测样本只是通过拉伸和压缩先验分布的曲面，让曲面更符合实际的参数分布，以表明参数的真实值最可能在哪里。

The data  𝑋  changes the surface of the space by pulling and stretching the fabric of the prior surface to reflect where the true parameters likely live. More data means more pulling and stretching, and our original shape becomes mangled or insignificant compared to the newly formed shape. Less data, and our original shape is more present. Regardless, the resulting surface describes the posterior distribution.
数据𝑋越多拉伸和压缩就越厉害，这样后验分布就变化的越厉害，可能完全看不出和先验分布的曲面有什么关系，或者说随着数据的增加先验分布对于后验分布的影响越来越小。这也体现了贝叶斯推断的核心思想：**你的先验应该尽可能合理，但是即使不是那么的合理也没有太大关系，MCMC会通过观测样本的拟合，尽可能将先验分布调整为符合观测样本的后验分布。**

但是如果数据𝑋较小，那么后验分布的形状会更多的反映出先验分布的形状。在小样本的情况下，MCMC对初始的先验分布会非常敏感。

**不同的先验概率对观测样本调整后验分布的阻力是不同的**

>**ANALOGY**: Probabilities allowed me to talk about quantum physics with you. Now Bayesian estimation will let me talk about Einstein's general relativity: Evidence warps the space of prior distributions much in the same way that planterary bodies warp space and create the force of gravity through geometry!  

![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200311230052.png)
时空曲率引起重力

For two dimensions, in the opposite of gravity, data essentially pushes up the original surface to make tall mountains. 

需要再次说明的是，在高维空间上，拉伸和挤压的变化难以可视化。在二维空间上，这些拉伸、挤压的结果是形成了几座山峰。而形成这些局部山峰的作用力会受到先验分布的阻挠。

The tendency of observed data to push up the posterior probability in certain areas is checked by the prior probability distribution, **so that small in magnitude prior probability means more resistance.**

先验分布越小意味着阻力越大；先验分布越大阻力越小。

>So priors and data likelhihood compete against each other, the same way your mind competes against two thoughts: "She smokes! I don't like her anymore!" competes with "but she is so pretty!"

有一点要特别注意，如果某处的先验分布为0，那么在这一点上也推不出后验概率。

Suppose the priors mentioned above represent different parameters  𝜆  of two Poisson distributions. Now, we observe a datapoint and visualize the new landscape. This datapoint is a random variate from a 2D Poisson distribution (think about the distribution as the number of emails you recieve and the number of text messages, in one day). This new data is going to change our priors into posteriors. It is going to warp the Bayesian fabric.
```python
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from scipy.stats.mstats import mquantiles
from separation_plot import separation_plot
import scipy.stats as stats

jet = plt.cm.jet
```
```python
# create observed data

# sample size of data we observe, trying varying this (keep it less than 100 ;)
N = 1

# the true parameters, but of course we do not see these values...
lambda_1_true = 1
lambda_2_true = 3

#...we only see the data generated, dependent on the above two values.
data = np.concatenate([
    stats.poisson.rvs(lambda_1_true, size=(N, 1)),
    stats.poisson.rvs(lambda_2_true, size=(N, 1))
], axis=1)

data
```
Now this is called the **data likelihood**, which in each dimension is the probability density function pdf (actually probability **mass** function pmf since our random variable is discrete rather than continuous) of the observed new data, which stems from a Poisson distribution, with expectation equal to the datapoint we observe (not the expectation of our priors, which is different):
```python
# plotting details.
x = y = np.linspace(.01, 5, 100)
likelihood_x = np.array([stats.poisson.pmf(data[:, 0], _x)
                        for _x in x]).prod(axis=1)
likelihood_y = np.array([stats.poisson.pmf(data[:, 1], _y)
                        for _y in y]).prod(axis=1)
L = np.dot(likelihood_x[:, None], likelihood_y[None, :])
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200311231745.png)

First, let's plot our priors: How we think the data will be distributed, before we actually observe any data. We start with **uniform** priors:

```python
uni_x = stats.uniform.pdf(x, loc=0, scale=5)
uni_y = stats.uniform.pdf(y, loc=0, scale=5)
M = np.dot(uni_x[:, None], uni_y[None, :])
im = plt.imshow(M, interpolation='none', origin='lower',
                cmap=jet, vmax=1, vmin=-.15, extent=(0, 5, 0, 5))
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title("Landscape formed by Uniform priors on $p_1, p_2$.")
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200311231950.png)

Our **posterior**, as given by Bayes' formula (we omit the denominator) is: The **data likelihood** L times the **prior** M. The mountain's peak is the datapoint we observe, because once we observe a datapoint, we expect all other datapoints to appear *at the same location*. We also plot, in yellow, the ***real*** expectation of the data generating mechanism, which we know is a Poisson with expectation (1, 3), and in green the datapoint we observe.
```python
plt.contour(x, y, M * L)
im = plt.imshow(M * L, interpolation='none', origin='lower',
                cmap=jet, extent=(0, 5, 0, 5))
plt.title("Landscape warped by %d data observation;\n Uniform priors on $p_1, p_2$." % N)
plt.scatter(lambda_2_true, lambda_1_true, c="y", s=50, edgecolor="none")
plt.xlim(0, 5)
plt.ylim(0, 5)
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200311232227.png)

Given the new datapoint, we now expect to observe data coming in with probability contours as pictured above: We expect new data to come in with highest probability exactly where the previous datapoint came in.

Now, what if we had started with **exponential** priors instead? In other words, we think there is a ***data fountain*** at the ***origin***, and we expected our data to come from there instead, rather than uniformly. Then our prior becomes:
```python
exp_x = stats.expon.pdf(x, loc=0, scale=3)
exp_y = stats.expon.pdf(x, loc=0, scale=10)
M = np.dot(exp_x[:, None], exp_y[None, :])

plt.contour(x, y, M)
im = plt.imshow(M, interpolation='none', origin='lower',
                cmap=jet, extent=(0, 5, 0, 5))
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title("Landscape formed by Exponential priors on $p_1, p_2$.")
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200311232426.png)

That is how we would expect data to flow if it flowed from the origin!

Now, we observe a datapoint! Yay! We're not alone in the universe anymore :-)

How does this datapoint modify our posterior M * L?
```python
plt.contour(x, y, M * L)
im = plt.imshow(M * L, interpolation='none', origin='lower',
                cmap=jet, extent=(0, 5, 0, 5))

plt.scatter(lambda_2_true, lambda_1_true, c="y", s=50, edgecolor="none")
plt.title("Landscape warped by %d data observation;\n Exponential priors on \
$p_1, p_2$." % N)
plt.xlim(0, 5)
plt.ylim(0, 5);
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200311232733.png)

We see that our datapoint created a mountain peak in our expectation fabric, just like before,but more *squished*. The top of that mountain is not the datapoint itself because our prior was a data fountain at the origin, it *fights* with our datapoint and the result of that tectonic shift is the landscape you see pictured above.

Let's bring this all together into a *combined* plot, and introduce more datapoints! It's like playing God with the tectonic plates on earth! 

We color in red our datapoints, in green the expectation which is nothing more than the $\lambda$s of our Poissons. You should be able to see that as you increase the number of datapoints, your posterior gets tighter and tighter (and taller and taller) and the peak gets closer and closer to the theoretical expecation, which is where the green point lies.

The more data we observe (from a theoretical Poission distribution will well know Expectation), the tighter and tighter our mountain centered around that expectation. We've already seen this phenomenon with coin tosses!
```python
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from scipy.stats.mstats import mquantiles
from separation_plot import separation_plot
import scipy.stats as stats

jet = plt.cm.jet

#
# create observed data
#

# sample size of data we observe, trying varying this (keep it less than 100 ;)
N = 1

# the true parameters, but of course we do not see these values...
lambda_1_true = 1
lambda_2_true = 3

#...we see the data generated, dependent on the above two values.
data = np.concatenate([
    stats.poisson.rvs(lambda_1_true, size=(N, 1)),
    stats.poisson.rvs(lambda_2_true, size=(N, 1))
], axis=1)
print(data)

#
# Likelihood function fron the observed data
#

x = y = np.linspace(.01, 5, 100)
likelihood_x = np.array([stats.poisson.pmf(data[:, 0], _x)
                        for _x in x]).prod(axis=1)
likelihood_y = np.array([stats.poisson.pmf(data[:, 1], _y)
                        for _y in y]).prod(axis=1)
L = np.dot(likelihood_x[:, None], likelihood_y[None, :])

#
# plots
#

figsize(12.5, 12)
plt.subplot(221)
uni_x = stats.uniform.pdf(x, loc=0, scale=5)
uni_y = stats.uniform.pdf(x, loc=0, scale=5)
M = np.dot(uni_x[:, None], uni_y[None, :])
im = plt.imshow(M, interpolation='none', origin='lower',
                cmap=jet, vmax=1, vmin=-.15, extent=(0, 5, 0, 5))
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title("Landscape formed by Uniform priors on $p_1, p_2$.")

plt.subplot(223)
plt.contour(x, y, M * L)
im = plt.imshow(M * L, interpolation='none', origin='lower',
                cmap=jet, extent=(0, 5, 0, 5))
plt.title("Landscape warped by %d data observation;\n Uniform priors on $p_1, p_2$." % N)
plt.scatter(lambda_2_true, lambda_1_true, c="g", s=50, edgecolor="none")
for _ in range(N):
    plt.scatter(data[_][0], data[_][1], c="r", s=50, edgecolor="none")
plt.xlim(0, 5)
plt.ylim(0, 5)

plt.subplot(222)
exp_x = stats.expon.pdf(x, loc=0, scale=3)
exp_y = stats.expon.pdf(x, loc=0, scale=10)
M = np.dot(exp_x[:, None], exp_y[None, :])

plt.contour(x, y, M)
im = plt.imshow(M, interpolation='none', origin='lower',
                cmap=jet, extent=(0, 5, 0, 5))
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.title("Landscape formed by Exponential priors on $p_1, p_2$.")

plt.subplot(224)
# This is the likelihood times prior, that results in the posterior.
plt.contour(x, y, M * L)
im = plt.imshow(M * L, interpolation='none', origin='lower',
                cmap=jet, extent=(0, 5, 0, 5))

plt.scatter(lambda_2_true, lambda_1_true, c="g", s=50, edgecolor="none")
for _ in range(N):
    plt.scatter(data[_][0], data[_][1], c="r", s=50, edgecolor="none")
plt.title("Landscape warped by %d data observation;\n Exponential priors on \
$p_1, p_2$." % N)
plt.xlim(0, 5)
plt.ylim(0, 5);
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200311234027.png)

The highest points on the mountain, corresponding the the darkest red, are biased towards (0,0) in the exponential case, which is the result from the exponential prior putting more prior weight in the (0,0) corner.

The green dot represents the true parameters, or **expectation**, and ***as we increase the number of datapoints, our mountain peak gets closer and closer to the expectation***.

四张图里的绿点代表参数的真实取值。每列的上图表示先验分布图形，下图表示后验分布图形。

我们主要到，虽然观测值相同，两种先验假设下得到的后验分布却有不同的图形。

我们从上图里注意到2个细节：

1. 右下方的指数先验对应的后验分布图形中，右上角区域的取值很低，原因是假设的指数先验在这一区域的取值也较低。
2. 另一方面，左下方的均匀分布对应的后验分布图形中，右上角区域的取值相对较高，这也是因为均匀先验在该区域的取值相比指数先验更高。
3. 在右下角指数分布的后验图形中，最高的山峰，也就是红色最深的地方，向（0，0）点偏斜，原因就是指数先验在这个角落的取值更高。

>思考：这个现象其实和深度学习里sigmoid函数的调整过程是类似的，sigmoid在越靠近0或1概率的区域中，调整的速率会越来越慢，即死胡同效应。因为这时候sigmoid会认为收敛已经接近尾声，要减缓调整的幅度，以稳定已经收敛到的结果。

Try changing the sample size to other values (try 5, 10, 100) and observe how the mountain posterior changes.

That is the essence of Bayesian dynamics.

### 使用 PyMC3 MCMC来搜索图景像

Exploring the deformed posterior space generated by our prior surface and observed data is a great exercise. However, we cannot naively search the space: Traversing $N$-dimensional space is exponentially difficult in $N$: the size of the space quickly blows-up as we increase $N$ ([the curse of dimensionality](http://en.wikipedia.org/wiki/Curse_of_dimensionality)).

How do we find these hidden mountains? The idea behind Markov Chain Monte Carlo algorithms is to perform an ***intelligent search*** of the space. 
搜索什么呢？一个点吗？肯定不是，贝叶斯思想的核心就是世界上没有100%确定的东西，所有的推断都是一个概率分布。

MCMC algorithms like **Metropolis** return **samples** from the posterior distribution, not the distribution itself. 

MCMC performs a task similar to repeatedly asking "*How likely is this pebble I found to be from the mountain I am searching for*?", and completes its task by returning thousands of accepted pebbles in hopes of reconstructing the original mountain. In MCMC and PyMC3 lingo, the returned sequence of "*pebbles*" are the **samples**, cumulatively called the **traces**. 
在MCMC和PyMC的术语里，这些返回序列里的“石头”就是观测样本，累计起来称之为“迹”。

我们希望MCMC搜索的位置能收敛到后验概率最高的区域（注意不是一个确定的点，是一个区域）。为此，MCMC每次都会探索附近位置上的概率值，并朝着概率值增加的方向前进。
MCMC does this by exploring nearby positions and moving into areas with higher probability, picking up samples from that area.

**Why do we pick up thousands of samples?**

我们可能会说，算法模型训练的目的不就是为了获得我们对随机变量的最优估计吗？毕竟在很多时候，我们训练得到的模型会用于之后的预测任务。但是贝叶斯学派不这么做，贝叶斯推断的结果更像是一个参谋，它只提供一个建议，而最后的决策需要我们自己来完成（例如我们通过取后验估计的均值作为最大后验估计的结果）

回到MCMC的训练返回结果，它返回成千上万的样本让人觉得这是一种低效的描述后验概率的方式。实际上这是一种非常高效的方法。下面是其它可能用于计算后验概率的方法

1. 用解析表达式描述“山峰区域”(后验分布)，这需要描述带有山峰和山谷的 N 维曲面。在数学上是比较困难的。
2. 也可以返回图形里的顶峰，这种方法在数学上是可行的，也很好理解（这对应了关于未知量的估计里最可能的取值），但是这种做法忽略了图形的形状，而我们知道，在一些场景下，这些形状对于判定未知变量的后验概率来说，是非常关键的
3. 除了计算原因，另一个主要原因是，利用返回的大量样本可以利用大数定理解决非常棘手问题。有了成千上万的样本，就可以利用直方图技术，重构后验分布曲面。

**MCMC algorithms**

There is a large family of algorithms that perform MCMC, not just Metropolis. Most of these algorithms can be expressed at a high level as follows:

1. Start at current position.
2. Propose moving to a new position (investigate a pebble near you).
3. Accept/Reject the new position based on the position's adherence to the data and prior distributions (ask if the pebble likely came from the mountain).
4.  A.  If you accept: Move to the new position. Return to Step 1.
    B. Else: Do not move to new position. Return to Step 1. 
5. After a large number of iterations, return all accepted positions.

This way we move in the general direction towards regions where the posterior distributions live, and collect samples sparingly on the journey. Once we reach the posterior distribution, we can easily collect samples as they likely all belong to the posterior distribution. 

If the current position of the MCMC algorithm is in an area of extremely low probability, which is often the case when the algorithm begins (typically at a random location in the space), the algorithm will move in positions *that are likely not from the posterior* but better than everything else nearby. Thus the first moves of the algorithm are not reflective of the posterior. That is why we often *throw away* the first samples.

In the above algorithm, notice that only the current position matters (new positions are investigated only near the current position). This property as *memorylessness*, i.e. the algorithm does not care *how* it arrived at its current position, only that it is there. That is why the chain is **Markovian**.

### 一个MCMC搜索图景像的实际的例子
Unsupervised Clustering using a Mixture Model（使用混合模型进行无监督聚类）
```python
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import scipy.stats as stats

jet = plt.cm.jet

figsize(12.5, 4)
data = np.loadtxt("mixture_data.csv", delimiter=",")

plt.hist(data, bins=20, color="b", histtype="stepfilled", alpha=0.8)
plt.title("Histogram of the dataset")
plt.ylim([0, None]);
print(data[:10], "...")
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200312001809.png)

It appears the data has a **bimodal** form, that is, it appears to have ***two peaks***, one near 120 and the other near 200. Perhaps there are *two clusters* within this dataset.
**聚类是一个很宽泛的概念，不一定仅限于我们所熟悉的欧式空间的kmeans，实际上，聚类不一定都是几何意义上的聚类。通过对原始数据集进行概率分布的拟合，从而获得数据集中每个点所属类别的概率分布，这也是一种聚类。**

**选择符合数据观测分布的数学模型**
1. For each data point, choose cluster 1 with probability $p$, else choose cluster 2. 
2. Draw a random variate from a Normal distribution with parameters $\mu_i$ and $\sigma_i$ where $i$ was chosen in step 1.
3. Repeat.

这个算法可以产生与观测数据相似的效果。所以选择这个算法作为模型。

但是现在的问题是我们不知道参数 𝑝 和正态分布的参数。所以要学习或者推断出这些未知变量。

用Nor0，Nor1分别表示正态分布。两个正态分布的参数都是未知的，参数分别表示为𝜇𝑖,𝜎𝑖，𝑖 = 0，1。