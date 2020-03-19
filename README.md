
# SIFT-BOW-CBIR




## Reference

* [How to efficiently find k-nearest neighbours in high-dimensional data? - StackOverflow](https://stackoverflow.com/questions/3962775/how-to-efficiently-find-k-nearest-neighbours-in-high-dimensional-data): kdtree is not efficient for search k-nearest neighborhood when the dimension of data is very high. But some approximate alternatives (which don't search strict k-nearest neighbors, but approximate ones) can be applied. Also, LSH is mentioned.

* [Is kdtree used for speeding k-means clustering or not? - StackOverflow](https://stackoverflow.com/questions/20587752/is-kdtree-used-for-speeding-k-means-clustering-or-not)

* [List attributes of an object - StackOverflow](https://stackoverflow.com/questions/2675028/list-attributes-of-an-object)


* [How to access object attribute given string corresponding to name of that attribute - StackOverflow](https://stackoverflow.com/questions/2612610/how-to-access-object-attribute-given-string-corresponding-to-name-of-that-attrib)


* [Is sift algorithm invariant in color? - StackOverflow](https://stackoverflow.com/questions/40694713/is-sift-algorithm-invariant-in-color): SIFT only processs grayscale image, if the input image is not, it will first convert input to grayscale image internally.


* [CBIR : Vocabulary Tree在10K图片的实验 - 知乎](https://zhuanlan.zhihu.com/p/20554144): A brief introduction and experiment on Vocabulary Tree. But while it's fast, I think vocabulary tree is still an approximate algorithm.

* [How to speed up KMeans from sklean - StackOverflow](https://stackoverflow.com/questions/46515481/how-to-speed-up-k-means-from-scikit-learn): MinibatchKMeans is an option.


* [Sift中尺度空间、高斯金字塔、差分金字塔（DOG金字塔）、图像金字塔 - CSDN博客](https://blog.csdn.net/dcrmg/article/details/52561656)


* [hcmarchezi/vocabulary_tree - github](https://github.com/hcmarchezi/vocabulary_tree/blob/master/voctree.ipynb): python vocabulary_tree from scratch.

* [snavely/VocabTree2 - github](https://github.com/snavely/VocabTree2): C++ vocabulary tree.


* [Python Bag of Words clustering - StackOverflow](https://stackoverflow.com/questions/33713939/python-bag-of-words-clustering): related to cv2.BOWKMeansTrainer, I don't know if this one is better compared to sklearns KMeans implementation.

* [kdtree算法-博客园](https://www.cnblogs.com/eyeszjwang/articles/2429382.html)


## Dev Notes

1. How to parallelize SIFT feature extraction in python?
The easiest way is using [joblib.Parallel](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html), however, if using defeault setting, the [following error](https://stackoverflow.com/questions/50615260/typeerror-cant-pickle-cv2-xfeatures2d-sift-objects-occured-during-using-jobli) may occur:

```
TypeError: can't pickle cv2.xfeatures2d_SIFT objects
```
I solve this problem by simply setting backend argument to 'threading':
```
cluster_des_list = Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self._aux_func_pt)(im_name)
                                                        for im_name in tqdm(cluster_im_names)
                                                       )
```
Here are also some other solutions I didn't try:

* [Multiprocessing with OpenCV and Python](https://www.pyimagesearch.com/2019/09/09/multiprocessing-with-opencv-and-python/)

* [toori67/pool_sift.py - gist](https://gist.github.com/toori67/3a79a302d98e34a19defb818001d3e33)

* [spacy with joblib library generates pickle.PicklingError: Could not pickle the task to send it to the workers - StackOverflow](https://stackoverflow.com/questions/56884020/spacy-with-joblib-library-generates-pickle-picklingerror-could-not-pickle-the)


2. Will [joblib.Parallel](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html) keep the order of data as if the code is executed searilly?

Since I'm matching image name by the row index of feature matrix, the order of image name list and feature matrix rows should keep the same. Fortunately, I found that, the oreder is kept well when I use joblib.Parallel. I used the [following snippet](https://github.com/joblib/joblib/issues/165) to confirm this fact:

```
from joblib import Parallel, delayed
import numpy as np
NUM = range(1000)
EXPECTED = [np.sqrt(x) for x in NUM]
for it in range(100):
    rnum = Parallel(n_jobs=-1, backend='threading')(delayed(np.sqrt)(x) for x in NUM)
    if not (rnum == EXPECTED):
        zped = zip(rnum, EXPECTED)
        print('Discrepancy in iteration %d' % (it))
        print([(x, ex) for (x, ex) in zped if x != ex])
        break
    else:
        print('Order kept.')
```

Note that, **I'm using joblib 0.14.1**. There are someone reported that, the result can be disordered, see the github issue [here](https://github.com/joblib/joblib/issues/165).

3. How to indicate the progress of feature extraction, especially in the multiprocessing block?
[tqdm](https://github.com/tqdm/tqdm) is superisingly powerful, and even works for multiprocessing scenario. The usage is also easy:

```
cluster_des_list = Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self._aux_func_pt)(im_name)
                                                        for im_name in tqdm(cluster_im_names)
                                                       )
```

Simply replacing `cluster_im_names` with `tqdm(cluster_im_names)` gives you a nice progressbar.

4. How to draw a bounding box of object detection on retrieved image?  
In short, this is achived by homography wrapping. Just wrap the source bounding box to retrieved image. You may refer to [How to draw bounding box on best matches?](https://stackoverflow.com/questions/51606215/how-to-draw-bounding-box-on-best-matches).
An official documents can be found here [Feature Matching + Homography to find Objects](https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html)

5. Which norm should be used for local descriptor matching?

```
it depends not on the images, but on the descriptors you use.

for binary descriptors, like ORB,BRIEF,BRISK you must use the HAMMING norm, the descriptors are bitstrings, not numbers

for float descriptors, like SURF or SIFT, use any of L1, L2, L2sqr (L2 probably works best)

the Feature2D class also has a defaultNorm() member, which you can query.
```
According to [Which norm is the best to match descriptors?](https://answers.opencv.org/question/147525/which-norm-is-the-best-to-match-descriptors/).
To match SIFT decriptor, if using HAMMING norm, there will be an error says:

```
error: OpenCV(3.4.2) /tmp/build/80754af9/opencv-suite_1535558553474/work/modules/core/src/batch_distance.cpp:245: error: (-215:Assertion failed) (type == 0 && dtype == 4) || dtype == 5 in function 'batchDistance'
```


## TODO

1. other way to compute similarity (especially asymmetric ones, e.g. KL-divergence)
1. test tfidf feature
2. vocabulary tree
2. use cv2 FLANN to replace sklearn KNN. See an example [here](https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html)
3. inverted index
4. support dynamically indexing new images
5. memory-thrift solution?
6. other feature support? (e.g. surf-bow, CNN feature)