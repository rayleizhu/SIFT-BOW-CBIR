
## SIFT-BOW-CBIR

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


