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

**1. 选择符合数据观测分布的数学模型**
1. For each data point, choose cluster 1 with probability $p$, else choose cluster 2. 
2. Draw a random variate from a Normal distribution with parameters $\mu_i$ and $\sigma_i$ where $i$ was chosen in step 1.
3. Repeat.

这个算法可以产生与观测数据相似的效果。所以选择这个算法作为模型。

但是现在的问题是我们不知道参数 𝑝 和正态分布的参数。所以要学习或者推断出这些未知变量。

用Nor0，Nor1分别表示正态分布。两个正态分布的参数都是未知的，参数分别表示为𝜇𝑖,𝜎𝑖，𝑖 = 0，1。
**2. 对模型的参数进行先验建模**

**所属类别分布先验**
Denote the Normal distributions $\text{N}_0$ and $\text{N}_1$. Both currently have *unknown* **mean** and **standard deviation**, denoted $\mu_i$ and $\sigma_i, \; i =0,1$ respectively. A specific data point can be from either $\text{N}_0$ or $\text{N}_1$, and we assume that the data point is assigned to $\text{N}_0$ with probability $p$, to $\text{N}_1$ with probability $1-p$.
对于某一个具体的数据点来说，它可能来自Nor0也可能来自Nor1， 假设数据点来自Nor0的概率为𝑝。 这是一个先验，由于我们并不知道来自 Nor1 的实际概率，因此我们只能用 0-1 上的均匀分布来进行建模假设（最大熵原理）。我们称该先验为 p。

有一种近似的方法，可以使用 PyMC 的类别(Categorical)随机变量将数据点分配给某一类别。PyMC 类别随机变量有一个𝑘维概率数组变量，必须对𝑘维概率数组变量进行 求和使其和变成 1，PyMC 类别随机变量的 value 属性是一个 0 到𝑘 − 1的值，该值如何选 择由概率数组中的元素决定(在本例中𝑘 = 2)。

*A priori*, we do not know what the probability of assignment to cluster 1 is, so we form a uniform variable on $(0, 1)$. We call call this $p_1$. The probability of belonging to cluster 2 is therefore $p_2 = 1 - p_1$. Note we should not use a normal variable, because that presupposes an expectation of 0.5, however in this case we ***have no expectation*** for each datapoint!
目前还不知道将数据分配给类别 1 的 先验概率是多少，所以选择 0 到 1 的均匀随机变量作为先验分布。此时输入类别变量的 概率数组为[𝑝, 1 − 𝑝]。

>**NOTE!!:**
>Unfortunately, we can't we just give `[p1, p2]` to our `Categorical` variable. PyMC3 uses `Theano` under the hood to build the models so we need to use `theano.tensor.stack()` to combine $p_1$ and $p_2$ into a vector that it can understand. We pass this vector into the `Categorical` variable as well as the `testval` parameter to give our variable an idea of where to start from: 300 random `0`s and `1`s for each of our 300 datapoints, indicating they belong to either cluster 0 or cluster 1.

```python
import pymc3 as pm
import theano.tensor as T

with pm.Model() as model:
    p1 = pm.Uniform('p', 0, 1)
    p2 = 1 - p1
    p = T.stack([p1, p2])
    
    # this produces worse results. Why? This looks like a categorical assignment with Bernoulli probability
    #assignment = pm.Categorical('assignment', np.array([0.5, 0.5]), 
    #                            shape = data.shape[0],
    #                            testval = np.random.randint(0, 2, data.shape[0]))
    
    # This is better: This looks like a categorical assignment with Dirichlet probabilty 
    assignment = pm.Categorical("assignment", p, 
                                shape = data.shape[0],
                                testval = np.random.randint(0, 2, data.shape[0]))
    
print("prior assignment, with p = %.2f:" % p1.tag.test_value)
print(assignment.tag.test_value[:100])
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200313114538.png)

**单个聚类中的正态分布的参数分布先验**
Looking at my dataset histogram, I would guess that the standard deviations of the two Normal distributions representing each cluster are different. To maintain ignorance of what the standard deviations might be, we will initially model them as uniform on 0 to 100. We will include both standard deviations in our model using a single line of PyMC3 code:

    sds = pm.Uniform("sds", 0, 100, shape=2)

Notice that we specified `shape=2`: we are modeling both $\sigma$s as a *single* PyMC3 variable. Note that this does not induce a necessary relationship between the two $\sigma$s, it is simply for succinctness. You could have picked two different variables.

We also need to specify priors on the *centers* of the clusters. The centers are really the $\mu$ parameters in these Normal distributions. Their priors can be modeled by a Normal distribution because, looking at the data, I have somewhat of an idea where the two centers might be &mdash; I would guess somewhere around 120 and 190 respectively, though I am not very confident in these eyeballed estimates. Hence I will set $\mu_0 = 120, \mu_1 = 190$ and $\sigma_0 = \sigma_1 = 10$.
（虽然是肉眼观察到的，但是从数据形状上来看，是在120和190附近，最重要的是：MCMC会帮助我们修正先验中不是那么精确的部分）

I will also assign a **deterministic** (the opposite of probabilistic) variable to each **probabilistic** `sds` and `centers` variable.

Let's do all this!

- `sds` is a Uniform distribution in[0, 100], in 2 dimensions (of shape 2)
- `centers` is a Normal distribution with mean 120 and standard deviation 10 in one dimension, and mean 120 and standard deviation 10 in the other dimension
- `center_i` is a (deterministic) array of **values** that changes with each `assignment` variable. When assignment is = 0, it denotes `centers[0]`, and when assignment is = 1, it denotes `centers[1]`. So the length of `center_i` is 300
- `sd_i` is a (deterministic) array of **values** that changes with each `assignment` variable. When assignment is = 0, it denotes `sds[0]`, and when assignment is = 1, it denotes `sds[1]`. So the length of `sd_i` is 300
- `observations` is a probabilistic Normal distribution that ***fits*** the observed `data`, with mean and standard deviation that changes for each datapoint in `data`. So sometimes it's `centers[0]` and `sds[0]`, and other times it's `centers[1]` and `sds[1]`, depending on the `assignment` value of each datapoint

```python
with model:
    sds = pm.Uniform("sds", 0, 100, shape=2)
    centers = pm.Normal("centers", 
                        mu=np.array([120, 190]), 
                        sd=np.array([10, 10]), 
                        shape=2)
    
    center_i = pm.Deterministic('center_i', centers[assignment])
    sd_i = pm.Deterministic('sd_i', sds[assignment])
    
    # and to combine it with the observations:
    observations = pm.Normal("obs", mu=center_i, sd=sd_i, observed=data)
    
print("Random assignments: ", assignment.tag.test_value[:100], "...")
print("Assigned center: ", center_i.tag.test_value[:100], "...")
print("Assigned standard deviation: ", sd_i.tag.test_value[:100])
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200313115830.png)

**3. MCMC搜索过程 - 迹**

Notice how we continue to build the model within the context of `Model()`. This automatically adds the variables that we create to our model. As long as we work within this context we will be working with the same variables that we have already defined. Any sampling that we do within the context of `Model()` will be done only on the model in which we are working. 

We will tell our model to explore the space that we have so far defined by defining the sampling methods, in this case **`Metropolis()` for our continuous variables and `ElemwiseCategorical()` for our categorical variable (Metropolis does not work for discrete variables).**

We will use these sampling methods to explore the space by using `sample(iterations, step)`, where `iterations` is the number of steps we wish the algorithm to perform and `step` is the way in which we want to handle those steps. 

We use our combination of `Metropolis()` and `ElemwiseCategorical()` for the `step` and sample 25,000 iterations:

>**Important to understand (repeat this 10 times)**: We *sample* from the posterior at every time step of our iteration, and this gives us different samples of all our model paramters. At each time step we get closer and closer to pdfs for our model parameters (`mu` and `sd`) that yield the data likelihood `observations` pdf which fits our empirical histogram the best. So when we converge, we have the exact modeling solution for our empirical data as model **distributions**, yeilding best point approximations as well as errors made (standard deviation).

>**NOTE**: We are going back in time to find the process that generated the data.
```python
with model:
    step1 = pm.Metropolis(vars=[p, sds, centers])
    step2 = pm.ElemwiseCategorical(vars=[assignment])
    trace = pm.sample(25000, step=[step1, step2])
```
>这一步我的mbp需要跑两分半

We have stored the paths of all our variables, or **traces**, in the `trace` variable. These paths are the routes the unknown parameters (centers, precisions, and $p$) have taken thus far in our exploration of our state space.

The individual path of each variable is indexed by the PyMC3 variable `name` that we gave that variable when defining it within our model. For example, `trace["sds"]` will return a `numpy array` object that we can then index and slice as we would any other `numpy array` object. 
```python
figsize(12.5, 9)
plt.subplot(311)
lw = 1
center_trace = trace["centers"]

# pretty colors
colors = ["#348ABD", "#A60628"] if center_trace[-1, 0] > center_trace[-1, 1] \
    else ["#A60628", "#348ABD"]

plt.plot(center_trace[:, 0], label="trace of center 0", c=colors[0], lw=lw)
plt.plot(center_trace[:, 1], label="trace of center 1", c=colors[1], lw=lw)
plt.title("Traces of unknown parameters")
leg = plt.legend(loc="upper right")
leg.get_frame().set_alpha(0.7)

plt.subplot(312)
std_trace = trace["sds"]
plt.plot(std_trace[:, 0], label="trace of standard deviation of cluster 0",
     c=colors[0], lw=lw)
plt.plot(std_trace[:, 1], label="trace of standard deviation of cluster 1",
     c=colors[1], lw=lw)
plt.legend(loc="upper left")

plt.subplot(313)
p_trace = trace["p"]
plt.plot(p_trace, label="$p$: frequency of assignment to cluster 0",
     color=colors[0], lw=lw)
plt.xlabel("Steps")
plt.ylim(0, 1)
plt.legend();
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200313164604.png)

从上图我们可以看出什么？

1. The traces converge, not to a single point, but to a *distribution* of possible points. This is *convergence* in an MCMC algorithm!
>这些迹并非收敛到某一点，而是收敛到一定分布下，概率较大的点集。这就是MCMC算法里收敛的涵义。

2. Inference using the first few thousand points is a bad idea, as they are unrelated to the final distribution we are interested in. Thus is it a good idea to discard those samples before using the samples for inference. We call this period before converge the *burn-in period*.
>最初的几千个点（训练轮）与最终的目标分布关系不大，所以使用这些点参与估计并不明智。一个聪明的做法是剔除这些点之后再执行估计，产生这些遗弃点的过程称为预热期。

3. The traces appear as a *random walk* around the space, that is, the paths exhibit correlation with previous positions. This is both good and bad. We will always have correlation between current positions and the previous positions, but too much of it means we are not exploring the space well. This will be detailed in the Diagnostics section later below.
>这些迹看起来像是在围绕空间中某一区域随机游走。这就是说它总是在基于之前的位置移动。这样的好处是确保了当前位置与之前位置之间存在直接、确定的联系。然而坏处就是太过于限制探索空间的效率。



To achieve further convergence, we will perform *more* MCMC steps. In the pseudo-code algorithm of MCMC above, the only position that matters is the current position (new positions are investigated near the current position), implicitly stored as part of the `trace` object. To continue where we left off, we pass the `trace` that we have already stored into the `sample()` function with the same step value. The values that we have already calculated will not be overwritten. This ensures that our sampling continues where it left off in the same way that it left off. 

We will sample the MCMC ***fifty thousand*** *more* times and visualize progress below:
```python
with model:
    trace = pm.sample(50000, step=[step1, step2], trace=trace)
```
```python
center_trace = trace["centers"][25000:]
prev_center_trace = trace["centers"][:25000]
center_trace.shape
```
```python
center_trace[:,1].shape
```
```python
figsize(12.5, 4)

x = np.arange(25000)
plt.plot(x, prev_center_trace[:, 0], label="previous trace of center 0",
     lw=lw, alpha=0.4, c=colors[1])
plt.plot(x, prev_center_trace[:, 1], label="previous trace of center 1",
     lw=lw, alpha=0.4, c=colors[0])

x = np.arange(25000, 75000)
plt.plot(x, center_trace[:50000, 0], label="new trace of center 0", lw=lw, c="#348ABD")
plt.plot(x, center_trace[:50000, 1], label="new trace of center 1", lw=lw, c="#A60628")

plt.title("Traces of unknown center parameters")
leg = plt.legend(loc="upper right")
leg.get_frame().set_alpha(0.8)
plt.xlabel("Steps");
```
**4. 如何估计各个未知变量的最佳后验估计值**

Our main challenge is to identify the clusters. 

We have determined posterior distributions for our unknowns. We plot the posterior distributions of the center and standard deviation variables below:
```python
figsize(11.0, 4)
std_trace = trace["sds"][25000:]
prev_std_trace = trace["sds"][:25000]

_i = [1, 2, 3, 4]
for i in range(2):
    plt.subplot(2, 2, _i[2 * i])
    plt.title("Posterior of center of cluster %d" % i)
    plt.hist(center_trace[:, i], color=colors[i], bins=30,
             histtype="stepfilled")

    plt.subplot(2, 2, _i[2 * i + 1])
    plt.title("Posterior of standard deviation of cluster %d" % i)
    plt.hist(std_trace[:, i], color=colors[i], bins=30,
             histtype="stepfilled")
    # plt.autoscale(tight=True)

plt.tight_layout()
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200313165910.png)

The MCMC algorithm has proposed that the most likely centers of the two clusters are near 120 and 200 respectively. The most likely standard deviation is 50 for cluster 0, and 22 for cluster 1.

We are also given the posterior distributions for the labels of the data point, which is present in `trace["assignment"]`. Below is a visualization of this. The y-axis represents a subsample of the posterior labels for each data point. The x-axis are the sorted values of the data points. A red square is an assignment to cluster 1, and a blue square is an assignment to cluster 0. 
```python
import matplotlib as mpl
figsize(10, 10)
plt.cmap = mpl.colors.ListedColormap(colors)
plt.imshow(trace["assignment"][::400, np.argsort(data)],
       cmap=plt.cmap, aspect=.4, alpha=.9)
plt.xticks(np.arange(0, data.shape[0], 40),
       ["%.2f" % s for s in np.sort(data)[::40]])
plt.ylabel("posterior sample")
plt.xlabel("value of $i$th data point")
plt.title("Posterior labels of data points");
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200313170110.png)
Looking at the above plot, it appears that the most uncertainty is between 150 and 170. The above plot slightly misrepresents things, as the x-axis is not a true scale (it displays the value of the $i$th sorted data point). A more clear diagram is below, where we estimate the *frequency* of each data point belonging to labels 0 and 1:
```python
figsize(10, 5)
cmap = mpl.colors.LinearSegmentedColormap.from_list("BMH", colors)
assign_trace = trace["assignment"]
plt.scatter(data, 1 - assign_trace.mean(axis=0), cmap=cmap,
        c=assign_trace.mean(axis=0), s=50)
plt.ylim(-0.05, 1.05)
plt.xlim(35, 300)
plt.title("Probability of data point belonging to cluster 0")
plt.ylabel("probability")
plt.xlabel("value of data point");
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200313170246.png)
Even though we modeled the clusters using Normal distributions, we didn't get just two Normal distributions that *best* fit the data, but a distribution of values for the Normal distributions' parameters. 

>Repeat this twice more, because it's *key* to Bayesian estimation.

可以看到，虽然对正态分布对两类数据进行了建模，MCMC也根据观测样本得到了未知变量的后验概率分布。但是我们仍然没有得到能够最佳匹配数据的正态分布，而仅仅是得到了关于正态分布各参数的分布。当然，这也体现了贝叶斯推断的一个特点，贝叶斯推断并不直接作出决策，它更多地是提供线索和证据，决策还是需要统计学家来完成。

那接下来一个很自然的问题是，我们如何能够选择能够满足最佳匹配的参数 - 均值、方差呢？

**一个简单粗暴的方法是选择后验分布的均值**（当然，这非常合理且有坚实的理论支撑）。在下图中，我们以后验分布的均值作为正态分布的各参数值，并将得到的正态分布于观测数据形状叠加到一起。
```python
norm = stats.norm
figsize(15, 5)
x = np.linspace(20, 300, 500)
posterior_center_means = center_trace.mean(axis=0)
posterior_std_means = std_trace.mean(axis=0)
posterior_p_mean = trace["p"].mean()

plt.hist(data, bins=20, histtype="step", normed=True, color="k",
     lw=2, label="histogram of data")
y = posterior_p_mean * norm.pdf(x, loc=posterior_center_means[0],
                                scale=posterior_std_means[0])
plt.plot(x, y, label="Cluster 0 (using posterior-mean parameters)", lw=3)
plt.fill_between(x, y, color=colors[1], alpha=0.3)

y = (1 - posterior_p_mean) * norm.pdf(x, loc=posterior_center_means[1],
                                      scale=posterior_std_means[1])
plt.plot(x, y, label="Cluster 1 (using posterior-mean parameters)", lw=3)
plt.fill_between(x, y, color=colors[0], alpha=0.3)

plt.legend(loc="upper left")
plt.title("Visualizing Clusters using posterior-mean parameters");
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200313170621.png)
通过结果可以看到，取均值作为后验比较好的“拟合”了观测数据

**5. 回到聚类：预测问题 - 到了该决策的时候了！**
The above clustering can be generalized to $k$ clusters. Choosing $k=2$ allowed us to visualize the MCMC better, and examine some very interesting plots. 

What about prediction? Suppose we observe a new data point, say $x = 175$, and we wish to label it to a cluster. It is foolish to simply assign it to the *closer* cluster center, as this ignores the standard deviation of the clusters, and we have seen from the plots above that this consideration is very important. More formally: we are interested in the *probability* (as we cannot be certain about labels) of assigning $x=175$ to cluster 1. Denote the assignment of $x$ as $L_x$, which is equal to 0 or 1, and we are interested in $P(L_x = 1 \;|\; x = 175 )$.  

A naive method to compute this is to re-run the above MCMC with the additional data point appended. The disadvantage with this method is that it will be slow to infer for each novel data point. Alternatively, we can try a *less precise*, but much quicker method. 

We will use Bayes Theorem for this:

$$ P( A | X ) = \frac{ P( X  | A )P(A) }{P(X) }$$

In our case, $A$ represents $L_x = 1$ and $X$ is the evidence we have: we observe that $x = 175$. For a particular sample set of parameters for our posterior distribution, $( \mu_0, \sigma_0, \mu_1, \sigma_1, p)$, we are interested in asking "*is the probability that $x$ is in cluster 1 **greater** than the probability it is in cluster 0*?", where the probability is dependent on the chosen parameters.

$$  P(L_x = 1| x = 175 ) \gt P(L_x = 0| x = 175 ) \;\;\;? \\\\[5pt]$$
$$  \frac{ P( x=175  | L_x = 1  )P( L_x = 1 ) }{P(x = 175) } \gt \frac{ P( x=175  | L_x = 0  )P( L_x = 0 )}{P(x = 175) } \;\;\;?$$

As the denominators are equal, they can be ignored (and good riddance, because computing the quantity $P(x = 175)$ can be difficult). 

$$  P( x=175  | L_x = 1  )P( L_x = 1 ) \gt  P( x=175  | L_x = 0  )P( L_x = 0 ) \;\;\;?$$

Let's write this equation down probabilistically, and look at its mean to get the most realistic estimate:
```python
norm_pdf = stats.norm.pdf
p_trace = trace["p"][25000:]
prev_p_trace = trace["p"][:25000]
x = 175

v = p_trace * norm_pdf(x, loc=center_trace[:, 0], scale=std_trace[:, 0]) > \
    (1 - p_trace) * norm_pdf(x, loc=center_trace[:, 1], scale=std_trace[:, 1])

print("Probability of belonging to cluster 1:", v.mean())
```
>('Probability of belonging to cluster 1:', 0.06006)

Giving us a probability instead of a label is a very useful thing. Instead of the naive 

    L = 1 if prob > 0.5 else 0

## MCMC收敛性讨论
### Using `MAP` to improve convergence

If you rerun the sims you may notice that our results are not consistent: Perhaps your initial cluster division was more scattered, or perhaps less scattered. The problem is that our traces are a function of the *starting values* of the MCMC algorithm. 即MCMC是初始值敏感的。这也很自然，MCMC的搜索过程是在做启发式搜索，类似于“盲人摸象”的过程，所以很自然地，不用的起点，其之后走的迹自然也是不同的。

It can be shown, mathematically, that letting the MCMC run long enough, by performing many steps, the algorithm *should forget its initial position*. In fact, this is what it means to say the MCMC converged (in practice though we can never achieve total convergence). 

Hence if we observe different posterior analysis, it is likely because our MCMC has not *fully converged yet*, and we should not use samples from it yet (we should use a larger burn-in period).

In fact, poor starting values can prevent any convergence, or significantly slow it down. Ideally, we would like to have the chain start at the *peak* of our landscape, as this is exactly where the posterior distributions exist. Hence, if we started at the peak, we could avoid a lengthy burn-in period and incorrect inference. Generally, we call this *peak* the [maximum a posterior](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) or, more simply, the *MAP*.
**我们常常会在深度学习项目中，直接基于resNet、googleNet这种已经经过训练优化后的模型，其背后的思想也有一些减少预热期的意思，在resNet、googleNet的基础上，在继续进行训练，可以更快地收敛到我们的目标概率分布上。**

Of course, we do not know where the MAP is. PyMC3 provides a function that will approximate, if not find, the MAP location. In the PyMC3 main namespace is the `find_MAP` function. If you call this function within the context of `Model()`, it will calculate the MAP which you can then pass to `pm.sample()` as a `start` parameter.

    start = pm.find_MAP()
    trace = pm.sample(2000, step=pm.Metropolis, start=start)

The `find_MAP()` function has the flexibility of allowing the user to choose which optimization algorithm to use (after all, this is a optimization problem: we are looking for the values that maximize our landscape), as not all optimization algorithms are created equal. 

The default optimization algorithm in function call is the Broyden-Fletcher-Goldfarb-Shanno ([BFGS](https://en.wikipedia.org/wiki/Broyden-Fletcher-Goldfarb-Shanno_algorithm)) algorithm to find the maximum of the log-posterior. 
As an alternative, you can use other optimization algorithms from the `scipy.optimize` module. For example, you can use Powell's Method, a favourite of PyMC blogger [Abraham Flaxman](http://healthyalgorithms.com/) [1], by calling `find_MAP(fmin=scipy.optimize.fmin_powell)`. 

### Diagnosing Convergence
**Autocorrelation-自相关（序列递归推演性）**

Autocorrelation is a measure of how related a series of numbers is with itself. A measurement of 1.0 is perfect positive autocorrelation, 0 no autocorrelation, and -1.0 is perfect negative correlation.  If you are familiar with standard *correlation*, then autocorrelation is just how correlated a series, $x_\tau$, at time $t$ is with the series at time $t-k$:

$$R(k) = Corr( x_t, x_{t-k} ) $$

>If the series is autocorrelated, you *can predict it*. If not, *you cannot*!

For example, consider the two series:

$$x_t \sim \text{Normal}(0,1), \;\; x_0 = 0$$
$$y_t \sim \text{Normal}(y_{t-1}, 1 ), \;\; y_0 = 0$$

which have example paths like:
```python
figsize(12.5, 4)

import pymc3 as pm
x_t = np.random.normal(0, 1, 200)
x_t[0] = 0
y_t = np.zeros(200)
for i in range(1, 200):
    y_t[i] = np.random.normal(y_t[i - 1], 1)

plt.plot(y_t, label="$y_t$", lw=3)
plt.plot(x_t, label="$x_t$", lw=3)
plt.xlabel("time, $t$")
plt.legend();
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200313221304.png)

One way to think of autocorrelation is **"*if I know the position of the series at time $s$, can it help me know where I am at time $t$*?"**

 In the series $x_t$, the answer is ***No***. By construction, $x_t$ are random variables. If I told you that $x_2 = 0.5$, could you give me a better guess about $x_3$? 
***No***.

On the other hand, $y_t$ is autocorrelated. By construction, if I knew that $y_2 = 10$, I can be very confident that $y_3$ will not be very far from 10. Similarly, I can even make a (less confident guess) about $y_4$: it will probably not be near 0 or 20, but a value of 5 is not too unlikely. I can make a similar argument about $y_5$, but again, I am less confident. Taking this to it's logical conclusion, we must concede that as $k$, the lag between time points, increases, the autocorrelation decreases. We can visualize this:
```python
def autocorr(x):
    # from http://tinyurl.com/afz57c4
    result = np.correlate(x, x, mode='full')
    result = result / np.max(result)
    return result[result.size // 2:]

colors = ["#348ABD", "#A60628", "#7A68A6"]

x = np.arange(1, 200)
plt.bar(x, autocorr(y_t)[1:], width=1, label="$y_t$",
        edgecolor=colors[0], color=colors[0])
plt.bar(x, autocorr(x_t)[1:], width=1, label="$x_t$",
        color=colors[1], edgecolor=colors[1])

plt.legend(title="Autocorrelation")
plt.ylabel("measured correlation \nbetween $y_t$ and $y_{t-k}$.")
plt.xlabel("k (lag)")
plt.title("Autocorrelation plot of $y_t$ and $x_t$ for differing $k$ lags.");
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200313221602.png)
Notice that as $k$ increases, the autocorrelation of $y_t$ decreases from a very high point. Compare with the autocorrelation of $x_t$ which looks like noise (***which it really is***), hence we can conclude no autocorrelation exists in this series. 
**还记得HMM隐马尔科夫假设吗？即当前的节点状态只和之前有限步骤（例如1步）的节点状态有关，虽然理论上应该是和历史上所有的节点状态有相关，但是其实越往前，相关性越小，甚至小到可以忽略，因为HMM的假设实际上并没有丢失太多的概率信息。**

**How does this relate to MCMC convergence?**

By the nature of the MCMC algorithm, we will always be returned samples that exhibit autocorrelation (this is because of the step `from your current position, move to a position near you`).

A chain that is ***not*** exploring the space well will exhibit very high autocorrelation. Visually, if the trace seems to meander like a river, and not settle down, the chain will have high autocorrelation.

This does not imply that a converged MCMC has low autocorrelation. Hence low autocorrelation is ***not necessary*** for convergence, but it is ***sufficient***.

>***If*** you observe low autocorrelation, your chain has converged. 

PyMC3 has a built-in autocorrelation plotting function in the `plots` module. 

**Thinning**

Another issue can arise if there is high-autocorrelation between posterior samples. Many post-processing algorithms require samples to be *independent* of each other. This can be solved, or at least reduced, by only returning to the user every $n$th sample, thus removing some autocorrelation. Below we perform an autocorrelation plot for $y_t$ with differing levels of **thinning**:
```python
max_x = 200 // 3 + 1
x = np.arange(1, max_x)

plt.bar(x, autocorr(y_t)[1:max_x], edgecolor=colors[0],
        label="no thinning", color=colors[0], width=1)
plt.bar(x, autocorr(y_t[::2])[1:max_x], edgecolor=colors[1],
        label="keeping every 2nd sample", color=colors[1], width=1)
plt.bar(x, autocorr(y_t[::3])[1:max_x], width=1, edgecolor=colors[2],
        label="keeping every 3rd sample", color=colors[2])

plt.autoscale(tight=True)
plt.legend(title="Autocorrelation plot for $y_t$", loc="lower left")
plt.ylabel("measured correlation \nbetween $y_t$ and $y_{t-k}$.")
plt.xlabel("k (lag)")
plt.title("Autocorrelation of $y_t$ (no thinning vs. thinning) \
at differing $k$ lags.");
```
![](https://markpersonal.oss-us-east-1.aliyuncs.com/pic/20200313222020.png)

With more thinning, the autocorrelation drops quicker. There is a tradeoff though: higher thinning requires more MCMC iterations to achieve the same number of returned samples. For example, 10 000 samples unthinned is 100 000 with a thinning of 10 (though the latter has less autocorrelation). 

What is a good amount of thinning? The returned samples will always exhibit some autocorrelation, regardless of how much thinning is done. So long as the autocorrelation tends to zero, you are probably ok. Typically thinning of more than 10 is not necessary.

Artificial Neural Network algorithms have a simliar concept, called **minibatch**:
- If one updates model parameters after processing the whole training data (i.e., epoch), it would take too long to get a model update in training, and the entire training data probably won’t fit in the memory.
- If one updates model parameters after processing every instance (i.e., stochastic gradient descent), model updates would be too noisy, and the process is not computationally efficient.
- Therefore, minibatch gradient descent is introduced as a trade-off between {fast model updates, memory efficiency} and {accurate model updates, computational efficiency}.

### `pymc3.plots`

It is not necessary to manually create histograms, autocorrelation plots and trace plots each time we perform MCMC. The authors of PyMC3 have included a visualization tool for just this purpose. 

The `pymc3.plots` module contains a few different plotting functions that you might find useful. For each different plotting function contained therein, you simply pass a `trace` returned from sampling as well as a list, `varnames`, of the variables that you are interested in. This module can provide you with plots of autocorrelation and the posterior distributions of each variable and their traces, among others.

Below we use the tool to plot the centers of the clusters.
```python
pm.plots.traceplot(trace, varnames=["centers"])
pm.plots.plot_posterior(trace["centers"][:,0])
pm.plots.plot_posterior(trace["centers"][:,1])
pm.plots.autocorrplot(trace, varnames=["centers"]);
```
The first plotting function gives us the posterior density of each unknown in the `centers` variable as well as the `trace` of each. `trace` plot is useful for inspecting that possible "*meandering*" property that is a result of non-convergence. The density plot gives us an idea of the shape of the distribution of each unknown, but it is better to look at each of them individually.

The second plotting function(s) provides us with a histogram of our model parameter samples with a few added features:
- The text overlay in the center shows us the posterior mean, which is a good summary of posterior distribution. 
- The interval marked by the horizontal black line overlay represents the *95% credible interval*, sometimes called the *highest posterior density interval* and not to be confused with a *95% confidence interval* (CI) of frequentist statistics. The *highest posterior density interval* can be interpreted as "*there is a 95% chance the parameter of interest lies in this interval*". When communicating your results to others, it is important to state this interval. 

One of the purposes for studying Bayesian methods is to have a clear understanding of our uncertainty in the model parameters. Combined with the posterior mean (the best "*guess*" for model parameters), the 95% credible interval provides a reliable interval to communicate the likely location of the best guess *and* the uncertainty (represented by the width of the interval).

> The *confidence interval* is a frequentist statistics construct. A confidence level represents the frequency (i.e. the proportion) of possible confidence intervals that contain the true value of the unknown population parameter. In other words, if confidence intervals are constructed using a given confidence level from an infinite number of independent sample statistics (parallel universes), the proportion of those intervals that contain the true value of the parameter will be equal to the confidence level.For example, if the confidence level (CL) is 95% then in hypothetical indefinite data collection, in 95% of the samples the interval estimate will contain the true population parameter. This is not the same as saying "*there is a 95% chance the parameter of interest lies in this interval*", and I much prefer the latter over the former!

The last plots, titled `center_0` and `center_1` are the generated autocorrelation plots, similar to the ones displayed above. Decreasing autocorrelation is a sufficient condition for convergence, so it gives us a good warm fuzzy we achieved convergence.

## Useful tips for MCMC

Bayesian inference would be the *de facto* method if it weren't for MCMC's computational difficulties (especially of the denominator). In fact, MCMC is what turns most users off practical Bayesian inference. Below are some good heuristics  to help convergence and speed up the MCMC engine. This is all part of the *art* of Bayesian estimation.

### Intelligent starting values

初始选择在后验概率附近，这样花很少的时间就可以计算出正确结果。We can aid the algorithm by telling where we *think* the posterior distribution will be by specifying the `testval` parameter in the `Stochastic` variable creation. In many cases we can produce a reasonable guess for the parameter. For example, if we have data from a Normal distribution, and we wish to estimate the $\mu$ parameter, then a good starting value would be the *mean* of the data. 

     mu = pm.Uniform( "mu", 0, 100, testval = data.mean() )

For most parameters in models, there is a frequentist estimate of it. These estimates are a good starting value for our MCMC algorithms. Of course, this is not always possible for some variables, but including as many appropriate initial values is always a good idea. Even if your guesses are wrong, the MCMC will still converge to the proper distribution, so there is little to lose.

This is what using `MAP` tries to do, by giving good initial values to the MCMC. So why bother specifying user-defined values? Well, even giving `MAP` good values will help it find the maximum a-posterior. 

Also important, *bad initial values* are a source of major bugs in PyMC3 and can hurt convergence.

### Good Priors

If the priors are poorly chosen, the MCMC algorithm may not converge, or at least have difficulty converging. Consider what may happen if the prior chosen does not even contain the true parameter: the prior assigns 0 probability to the unknown, hence the posterior will assign 0 probability as well. This can cause pathological results.

For this reason, it is best to *carefully* choose the priors. Often, lack of covergence or evidence of samples crowding to boundaries implies something is wrong with the chosen priors (see *Folk Theorem of Statistical Computing* below). 

### Covariance matrices and eliminating parameters

Minimizing the number of parameters in your model and especially parameters that are interdependent is an important consideration. You can use covariance matrices to see if parameters turn out to be very dependent. If so, return to the drawing board and build a lower-dimensional model with statistically independent parameters.

### The Folk Theorem of Statistical Computing

>   *If you are having computational problems Ilike getting many `NaN`s, probably your model is wrong.*
