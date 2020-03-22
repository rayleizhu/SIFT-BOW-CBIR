import numpy as np
import cv2
import sklearn
import joblib
import random
import copy
import numpy as np
import os
from sklearn import cluster, neighbors, feature_extraction

from collections import OrderedDict
import pandas as pd

# multiprocessing
from joblib import Parallel, delayed
import multiprocessing

# logging
from tqdm import tqdm
import logging

# customized utils
from utils import read_and_crop

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


SEED=10086
np.random.seed(SEED)
random.seed(SEED)



class ImageDB:
    def __init__(self, im_fea_name='bow',
                 pt_fea_name='sift',
                 vsize=4096,
                 top_k=50,
                 n_jobs=-1,
                 metric=None):
        self.im_fea_support = ['bow', 'tfidf']
        self.pt_fea_support = ['sift']
        assert im_fea_name in self.im_fea_support
        assert pt_fea_name in self.pt_fea_support
        self.im_fea_name = im_fea_name
        self.pt_fea_name = pt_fea_name
        self.vsize = vsize
        self.top_k = top_k
        self.n_jobs = n_jobs
        
        if pt_fea_name == 'sift':
            self.pt_fea_extractor = cv2.xfeatures2d.SIFT_create()
        else:
            raise NotImplementedError("Pont feature extraction method {:s} is not implemented!".format(pt_fea_name))
         
        if im_fea_name == 'tfidf':
            self.tfidf_transformer = sklearn.feature_extraction.text.TfidfTransformer(norm='l1')
            
        if metric is None:
            self.metric = 'cosine'
        else:
            self.metric = metric
            
            
            
        self.num_imgs = None
        
        self.kmeans = None
        self.knn = None
        
        self.im_dir = None
        # im_name_list contains all image names in database dir
        self.im_name_list = None
        # im_fea_mat is a matrix of shape (num_imgs, vsize)
        self.im_fea_mat = None
        
          
    def build(self, img_db_dir, cluster_im_num=1.0):
        assert cluster_im_num > 0.
        self.im_dir = img_db_dir
        self.im_name_list = sorted(os.listdir(img_db_dir))
        self.num_imgs = len(self.im_name_list)
        logging.info('{:d} images have been found in database!'.format(self.num_imgs))
        
        cluster_im_names = copy.deepcopy(self.im_name_list)
        random.shuffle(cluster_im_names)
        if cluster_im_num <= 1.0:
            cluster_im_num = int(cluster_im_num * self.num_imgs)
        else:
            cluster_im_num = int(cluster_im_num)
        cluster_im_names = cluster_im_names[:cluster_im_num]
        
        logging.info('Extracting pixel-level features using {:d} images...'.format(cluster_im_num))
#         cluster_des_list = []
#         for im_name in cluster_im_names:
#             im_path = os.path.join(self.im_dir, im_name)
#             im_bgr = cv2.imread(im_path)
#             kp, des = self._compute_pt_fea(im_bgr)
#             # kp cannot be serialized using pickle, we just simply throw it
#             cluster_des_list.append(des)
        cluster_des_list = Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self._aux_func_pt)(im_name)
                                                        for im_name in tqdm(cluster_im_names)
                                                       )
        des_mat_all = np.concatenate(cluster_des_list, axis=0)
        logging.info('{:d} key points have been extracted!'.format(len(des_mat_all)))
              
       # do clustering
        logging.info('Running KMeans clustering with {:d} centers...'.format(self.vsize))
        # we use MiniBatchKMeans since it can handle large datasets
        self.kmeans = cluster.MiniBatchKMeans(n_clusters=self.vsize,
                                              init_size=10*self.vsize,
                                              batch_size=self.vsize,
                                              random_state=SEED)
        self.kmeans.fit(des_mat_all)
        
        logging.info('Building {:d}*{:d} image-level feature matrix...'.format(self.num_imgs, self.vsize))
        # build bow feature matrix for images in database
#         self.im_fea_mat = []
#         for im_name in self.im_name_list:
#             im_path = os.path.join(self.im_dir, im_name)
#             im_bgr = cv2.imread(im_path)
#             kp, des = self._compute_pt_fea(im_bgr)
#             cur_hist = self._compute_im_bow(des)
#             self.im_fea_mat.append(cur_hist)
        self.im_fea_mat = Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self._aux_func_im)(im_name)
                                                       for im_name in tqdm(self.im_name_list)
                                                      )
        self.im_fea_mat = np.stack(self.im_fea_mat, axis=0)
        logging.debug('self.im_fea_mat.shape: {:s}'.format(str(self.im_fea_mat.shape)))
                
        if self.im_fea_name == 'tfidf':
            self.im_fea_mat = self.tfidf_transformer.fit_transform(self.im_fea_mat)
            
        logging.info('Building KNN for image retrieval...')
        # to search most similar images, here we use sklearn's KNN
        self.knn = neighbors.NearestNeighbors(n_neighbors=self.top_k,
                                              n_jobs=self.n_jobs,
                                              metric=self.metric)
        self.knn.fit(self.im_fea_mat)
        logging.info('Building Finished!')

        
    def retrieve(self, query_img, top_k=50):
        if top_k is None:
            top_k = self.top_k
        kp, des = self._compute_pt_fea(query_img)
        img_fea = np.expand_dims(self._compute_im_bow(des), axis=0)
        if self.im_fea_name == 'tfidf':
            img_fea = self.tfidf_transformer.transform(img_fea)
        dists, inds = self.knn.kneighbors(img_fea, top_k, return_distance=True)
        similar_imgs = []
        for cnt, im_idx in enumerate(inds[0]):
            similar_im_path = os.path.join(self.im_dir, self.im_name_list[im_idx])
            similar_imgs.append((similar_im_path, dists[0][cnt]))
        return similar_imgs
    
    
    def retrieve_and_locate(self, query_im_path, query_bb_path, top_k=10, match_num_for_hom=20, locate=True):
        if top_k is None:
            top_k = self.top_k
            
        # read and crop image
        cropped_patches, query_img, query_bbs = read_and_crop(query_im_path, query_bb_path)
        
        # compute local features
        kp_des_list = [ self._compute_pt_fea(patch) for patch in cropped_patches ]
        des_mat_cat = np.concatenate([ tp[1] for tp in kp_des_list ], axis=0)
        
        # compute image-level features
        img_fea = np.expand_dims(self._compute_im_bow(des_mat_cat), axis=0)
        if self.im_fea_name == 'tfidf':
            img_fea = self.tfidf_transformer.transform(img_fea)
        logging.debug('type(img_fea): {:s}'.format(str(type(img_fea))))
            
         # search similar images
        dists, inds = self.knn.kneighbors(img_fea, top_k, return_distance=True)
        similar_imgs = []
        for cnt, im_idx in enumerate(inds[0]):
            similar_im_path = os.path.join(self.im_dir, self.im_name_list[im_idx])
            similar_imgs.append((similar_im_path, dists[0][cnt]))
            
        if locate:
            result = OrderedDict()
            result['query_img'] = (query_im_path, 0.0, query_bbs)
            # locate objects in similar images
            # shouldn't use cv2.NORM_HAMMING for SIFT matching
            # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            if self.pt_fea_name in ['sift', 'surf']:
                norm = cv2.NORM_L2
            else: # ORB, BRIEF, BRISK
                norm = cv2.NORM_HAMMING
            bf = cv2.BFMatcher(norm, crossCheck=True)
            for idx, (img_sim_path, score) in enumerate(similar_imgs):
                bbs = []
                img_sim = cv2.imread(img_sim_path)
                # kp, des in retrieved image
                kp_sim, des_sim = self._compute_pt_fea(img_sim)
    #             # kp, des in query image
    #             kp_obj = []
    #             for kp, des in kp_des_list:
    #                 kp_obj += kp
    #             des_obj = des_mat_cat
    #             logging.debug('des_obj.shape: {:s}'.format(str(des_obj.shape)))
    #             logging.debug('des_sim.shape: {:s}'.format(str(des_sim.shape)))

    #             # match key point
    #             matches = bf.match(des_obj, des_sim)
    #             matches = sorted(matches, key = lambda x:x.distance)

    #             # only use good matches for homography estimation
    #             match_num = len(matches)
    #             if match_num < match_num_for_hom:
    #                 logging.warn('Only {:d} matches have been found, lower than match_num_for_hom {:d} you set.'.format(match_num , match_num_for_hom))
    #                 match_num_for_hom = match_num 
    #             good_matches = matches[:match_num_for_hom]

    #             # estimate homography matrix
    #             src_pts = np.float32([ kp_obj[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    #             dst_pts = np.float32([ kp_sim[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    #             M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    #             for i in range(len(cropped_patches)):
                for i, (kp_obj, des_obj) in enumerate(kp_des_list):
                    logging.debug('des_obj.shape: {:s}'.format(str(des_obj.shape)))
                    logging.debug('des_sim.shape: {:s}'.format(str(des_sim.shape)))
                    # match key point
                    matches = bf.match(des_obj, des_sim)
                    matches = sorted(matches, key = lambda x:x.distance)
                    # only use good matches for homography estimation
                    match_num = len(matches)
                    if match_num < match_num_for_hom:
                        logging.warn('Only {:d} matches have been found, lower than match_num_for_hom {:d} you set.'.format(match_num , match_num_for_hom))
                        match_num_for_hom = match_num 
                    good_matches = matches[:match_num_for_hom]
                    # estimate homography matrix
                    src_pts = np.float32([ kp_obj[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
                    dst_pts = np.float32([ kp_sim[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    # homograpy wrapping for corner points
                    h, w = cropped_patches[i].shape[:2]
                    src_corners = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    dst_corners = cv2.perspectiveTransform(src_corners, M)
                    bbs.append(dst_corners)

                key = 'top {:d} similar'.format(idx) 
                result[key] = (img_sim_path, score, bbs)
            return result
        return similar_imgs
    
    
    def migrate_to_tfidf(self):
        assert self.im_fea_name in ['bow', 'tfidf'], \
               'current image-level feature is {:s}'.format(self.im_fea_name)
        self.im_fea_name = 'tfidf'
        
        logging.info('Normalizing BOW to TD-IDF ...')
        self.tfidf_transformer = sklearn.feature_extraction.text.TfidfTransformer(norm='l1')
        self.im_fea_mat = self.tfidf_transformer.fit_transform(self.im_fea_mat)
        
        logging.info('Rebuilding KNN for image retrieval...')
        # to search most similar images, here we use sklearn's KNN
        self.knn = neighbors.NearestNeighbors(n_neighbors=self.top_k,
                                              n_jobs=self.n_jobs,
                                              metric=self.metric)
        self.knn.fit(self.im_fea_mat)
        logging.info('Building Finished!')

    
    def switch_sim_metric(self, metric):
        '''
        switch similarity metric for image retrieval
        args:
            metric: string or callable
        '''
        self.metric = metric
        logging.info('Rebuilding KNN for image retrieval...')
        # to search most similar images, here we use sklearn's KNN
        self.knn = neighbors.NearestNeighbors(n_neighbors=self.top_k,
                                              n_jobs=self.n_jobs,
                                              metric=self.metric)
        self.knn.fit(self.im_fea_mat)
        logging.info('Building Finished!')
        
    
    def _compute_pt_fea(self, im_bgr):
        '''
        args:
            img: cv2 BGR image
        return:
            (kp, des): kp is a list containing n feature point coordinate (cv2.Keypoint object),
                       des is a matrix of shape (n, 128)
        '''
         # sift is an algorithm for grayscale image, here we do conversion explicitly
         # https://stackoverflow.com/a/41290545
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.pt_fea_extractor.detectAndCompute(im_gray, None)
        return kp, des
    
    
    def _compute_im_bow(self, pt_fea_mat):
        '''
        args:
            pt_fea_mat: a descriptor matrix of shape (n, 128)
        return:
            hist: a histogram of visual word frequency, shape (vsize, )
        '''
        num_fea_pt = pt_fea_mat.shape[0]
        word_idx = self.kmeans.predict(pt_fea_mat)
#         hist = np.zeros(self.vsize)
#         for pt_idx in range(num_fea_pt):
#             hist[word_idx[pt_idx]] += 1
        hist = np.bincount(word_idx.ravel(), minlength=self.vsize)
        return hist
    
    
    def _aux_func_pt(self, im_name):
        # this function is created to speed up point_level feature extraction using joblib
        # see self.build()
        im_path = os.path.join(self.im_dir, im_name)
        im_bgr = cv2.imread(im_path)
        kp, des = self._compute_pt_fea(im_bgr)
        return des
    
    
    def _aux_func_im(self, im_name):
        # this function is created to speed up image_level feature extraction using joblib
        # see self.build()
        im_path = os.path.join(self.im_dir, im_name)
        im_bgr = cv2.imread(im_path)
        kp, des = self._compute_pt_fea(im_bgr)
        cur_hist = self._compute_im_bow(des)
        return cur_hist
    
    
    def save(self, path):
        # sift detector cannot be serialized, simply skip this attribute
        skip_key = ['pt_fea_extractor']
        state_dict = {}
        for k, v in self.__dict__.items():
            if k not in skip_key:
                state_dict[k] = v
        joblib.dump(state_dict, path)
        
        
    def load(self, path):
        # may partially load necessary attributes
        state_dict = joblib.load(path)
        for attrname, value in state_dict.items():
            setattr(self, attrname, value)
        
        
    def __len__(self):
        return self.num_imgs
        
        