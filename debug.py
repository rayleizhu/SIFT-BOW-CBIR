from image_db_v2 import ImageDB

cluster_im_num = 2048
print(cluster_im_num)
vsize = 2048
im_dir = '/home/zhulei/Data/pg_data/Images'
save_name = 'cin{:d}_vsize{:d}_db.joblib'.format(cluster_im_num, vsize)

im_db_all = ImageDB(vsize=vsize)
im_db_all.build(im_dir, cluster_im_num)
im_db_all.save(save_name)