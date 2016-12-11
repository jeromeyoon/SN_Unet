from ops import *
import tensorflow as tf

class networks(object):
    def __init__(self,batch_size,df_dim):
	self.batch_size = batch_size
	self.df_dim = df_dim
    def generator(self,nir,keep_prob):
	#### encoder ####
        enc0 = lrelu(conv2d(nir,self.df_dim,name='g_enc0'))
	g_bn1 = batch_norm(self.batch_size,name='g_bn1')
        enc1 =lrelu(g_bn1(conv2d(enc0,self.df_dim*2,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc1'))) #output:128x128
	g_bn2 = batch_norm(self.batch_size,name='g_bn2')
        enc2 =lrelu(g_bn2(conv2d(enc1,self.df_dim*2,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc2'))) #output:64x64
	g_bn3 = batch_norm(self.batch_size,name='g_bn3')
        enc3 =lrelu(g_bn3(conv2d(enc2,self.df_dim*4,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc3'))) #output:32x32
	g_bn4 = batch_norm(self.batch_size,name='g_bn4') 
        enc4 =lrelu(g_bn4(conv2d(enc3,self.df_dim*4,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc4'))) #output:16x16
	g_bn5 = batch_norm(self.batch_size,name='g_bn5')
        enc5 =lrelu(g_bn5(conv2d(enc4,self.df_dim*4,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc5'))) #output:8x8
	#### Deconder #####
	g_bn8 = batch_norm(self.batch_size,name='g_bn8')
	dec1_1 = g_bn8(deconv2d(enc5,[self.batch_size,16,16,self.df_dim*4],name='g_dec1_1'))
	"""
	dec1 = tf.image.resize_nearest_neighbor(enc5,[16,16])
	dec1 = tf.pad(dec1,[[0,0],[1,1],[1,1],[0,0]])
	dec1_1 = tf.nn.dropout(g_bn8(conv2d(dec1,self.df_dim*4,k_h=3,k_w=3,padding='VALID',name='g_dec1_1')),keep_prob) #output 16x16
	"""
	dec1_2 = tf.nn.tanh(conv2d(dec1_1,3,k_h=5,k_w=5,padding='SAME',name='g_dec1_2'))
	dec1_3 = lrelu(tf.concat(3,[dec1_1,enc4,dec1_2]))

	g_bn9 = batch_norm(self.batch_size,name='g_bn9')
	"""
	dec2 = tf.image.resize_nearest_neighbor(dec1_3,[32,32])
	dec2 = tf.pad(dec2,[[0,0],[1,1],[1,1],[0,0]])
	dec2_1 = tf.nn.dropout(g_bn9(conv2d(dec2,self.df_dim*4,k_h=3,k_w=3,padding='VALID',name='g_dec2_1')),keep_prob) #output 32x32
	"""
	dec2_1 = g_bn9(deconv2d(dec1_3,[self.batch_size,32,32,self.df_dim*4],name='g_dec2_1'))
	dec2_2 = tf.nn.tanh(conv2d(dec2_1,3,k_h=3,k_w=3,padding='SAME',name='g_dec2_2'))
	dec2_3 = lrelu(tf.concat(3,[dec2_1,enc3,dec2_2]))

	g_bn10 = batch_norm(self.batch_size,name='g_bn10')
	"""
	dec3 = tf.image.resize_nearest_neighbor(dec2_3,[64,64])
	dec3 = tf.pad(dec3,[[0,0],[1,1],[1,1],[0,0]])
	dec3_1 = g_bn10(conv2d(dec3,self.df_dim*4,k_h=3,k_w=3,padding='VALID',name='g_dec3_1')) #output 64x64
	"""
	dec3_1 = g_bn10(deconv2d(dec2_3,[self.batch_size,64,64,self.df_dim*4],name='g_dec3_1'))
	dec3_2 = tf.nn.tanh(conv2d(dec3_1,3,k_h=3,k_w=3,padding='SAME',name='g_dec3_2'))
	dec3_3 = lrelu(tf.concat(3,[dec3_1,enc2,dec3_2]))

	g_bn11 = batch_norm(self.batch_size,name='g_bn11')
	"""
	dec4 = tf.image.resize_nearest_neighbor(dec3_3,[128,128])
	dec4 = tf.pad(dec4,[[0,0],[1,1],[1,1],[0,0]])
	dec4_1 = g_bn11(conv2d(dec4,self.df_dim*2,k_h=3,k_w=3,padding='VALID',name='g_dec4_1')) #output 128x128
	"""
	dec4_1 = g_bn11(deconv2d(dec3_3,[self.batch_size,128,128,self.df_dim*2],name='g_dec4_1'))
	dec4_2 = tf.nn.tanh(conv2d(dec4_1,3,k_h=3,k_w=3,padding='SAME',name='g_dec4_2'))
	dec4_3 = lrelu(tf.concat(3,[dec4_1,enc1,dec4_2]))

	g_bn12 = batch_norm(self.batch_size,name='g_bn12')
	"""
	dec5 = tf.image.resize_nearest_neighbor(dec4,[256,256])
	dec5 = tf.pad(dec5,[[0,0],[1,1],[1,1],[0,0]])
	dec5_1 = g_bn12(conv2d(dec5,self.df_dim,k_h=3,k_w=3,padding='VALID',name='g_dec5_1')) #output 256x256
	"""
	dec5_1 = g_bn12(deconv2d(dec4_3,[self.batch_size,256,256,self.df_dim],name='g_dec5_1'))
	dec5_2 = lrelu(tf.concat(3,[dec5_1,enc0]))
	dec6 = conv2d(dec5_2,3,k_h=1,k_w=1,name='g_dec6') #output 256x256
	
	return dec1_2,dec2_2,dec3_2,dec4_2, tf.nn.tanh(dec6)

    def discriminator(self, image,keep_prob, reuse=False):
	if reuse:
            tf.get_variable_scope().reuse_variables()    
        h0 = lrelu(conv2d(image, self.df_dim, k_h=4,k_w=4,d_h=2,d_w=2,name='d_h0_conv')) #output size 128 x128
	d_bn1 = batch_norm(self.batch_size,name='d_bn1')
        h1 = lrelu(d_bn1(conv2d(h0, self.df_dim*2, k_h=4,k_w=2,d_h=2,d_w=2,name='d_h1_conv'))) #output size64 x64
	d_bn2 = batch_norm(self.batch_size,name='d_bn2')
        h2 = lrelu(d_bn2(conv2d(h1, self.df_dim*4, k_h=4,k_w=4,d_h=2,d_w=2,name='d_h2_conv'))) #output size 32 x32
	d_bn3 = batch_norm(self.batch_size,name='d_bn3')
        h3 = lrelu(d_bn3(conv2d(h2, self.df_dim*8,k_h=4,k_w=4,d_h=2,d_w=2,name='d_h3_conv'))) #output size 16x16
	d_bn4 = batch_norm(self.batch_size,name='d_bn4')
        h4 = lrelu(d_bn4(conv2d(h3,self.df_dim*8,k_w=4,k_h=4,d_h=2,d_w=2,name='d_h4_conv'))) #output size 8x 8
	d_bn5 = batch_norm(self.batch_size,name='d_bn5')
        h5 = lrelu(d_bn5(conv2d(h4,self.df_dim*8,k_w=4,k_h=4,d_h=2,d_w=2,name='d_h5_conv'))) #output size 4x4
        h6 = lrelu(linear(tf.reshape(h5, [self.batch_size, -1]), 1024, 'd_h6_lin'))
	h6 = tf.nn.dropout(h6,keep_prob)
        h7 = linear(h6, 1, 'd_h7_lin')
	
        return tf.nn.sigmoid(h7) 
    
    def sampler(self,nir,keep_prob):
	
	tf.get_variable_scope().reuse_variables()

	#### encoder ####
        enc0 = lrelu(conv2d(nir,self.df_dim,name='g_enc0'))
	g_bn1 = batch_norm(self.batch_size,name='g_bn1')
        enc1 =lrelu(g_bn1(conv2d(enc0,self.df_dim*2,k_h=3,k_w=3,d_h=2,d_w=2,name='g_enc1'),train=False)) #output:128x128
	g_bn2 = batch_norm(self.batch_size,name='g_bn2')
        enc2 =lrelu(g_bn2(conv2d(enc1,self.df_dim*2,k_h=3,k_w=3,d_h=2,d_w=2,name='g_enc2'),train=False)) #output:64x64
	g_bn3 = batch_norm(self.batch_size,name='g_bn3')
        enc3 =lrelu(g_bn3(conv2d(enc2,self.df_dim*4,k_h=3,k_w=3,d_h=2,d_w=2,name='g_enc3'),train=False)) #output:32x32
	g_bn4 = batch_norm(self.batch_size,name='g_bn4') 
        enc4 =lrelu(g_bn4(conv2d(enc3,self.df_dim*4,k_h=3,k_w=3,d_h=2,d_w=2,name='g_enc4'),train=False)) #output:16x16
	g_bn5 = batch_norm(self.batch_size,name='g_bn5')
        enc5 =lrelu(g_bn5(conv2d(enc4,self.df_dim*4,k_h=3,k_w=3,d_h=2,d_w=2,name='g_enc5'),train=False)) #output:8x8
	#### Deconder #####
	g_bn8 = batch_norm(self.batch_size,name='g_bn8')
	dec1 = tf.image.resize_nearest_neighbor(enc5,[16,16])
	dec1 = tf.pad(dec1,[[0,0],[1,1],[1,1],[0,0]])
	dec1_1 = tf.nn.dropout(g_bn8(conv2d(dec1,self.df_dim*4,k_h=3,k_w=3,padding='VALID',name='g_dec1_1'),train=False),keep_prob) #output 16x16
	dec1_2 = tf.nn.tanh(conv2d(dec1_1,3,k_h=5,k_w=5,padding='SAME',name='g_dec1_2'))
	dec1_3 = lrelu(tf.concat(3,[dec1_1,enc4,dec1_2]))

	g_bn9 = batch_norm(self.batch_size,name='g_bn9')
	dec2 = tf.image.resize_nearest_neighbor(dec1_3,[32,32])
	dec2 = tf.pad(dec2,[[0,0],[1,1],[1,1],[0,0]])
	dec2_1 = tf.nn.dropout(g_bn9(conv2d(dec2,self.df_dim*4,k_h=3,k_w=3,padding='VALID',name='g_dec2_1'),train=False),keep_prob) #output 32x32
	dec2_2 = tf.nn.tanh(conv2d(dec2_1,3,k_h=3,k_w=3,padding='SAME',name='g_dec2_2'))
	dec2_3 = lrelu(tf.concat(3,[dec2_1,enc3,dec2_2]))

	g_bn10 = batch_norm(self.batch_size,name='g_bn10')
	dec3 = tf.image.resize_nearest_neighbor(dec2_3,[64,64])
	dec3 = tf.pad(dec3,[[0,0],[1,1],[1,1],[0,0]])
	dec3_1 = g_bn10(conv2d(dec3,self.df_dim*4,k_h=3,k_w=3,padding='VALID',name='g_dec3_1'),train=False) #output 64x64
	dec3_2 = tf.nn.tanh(conv2d(dec3_1,3,k_h=3,k_w=3,padding='SAME',name='g_dec3_2'))
	dec3_3 = lrelu(tf.concat(3,[dec3_1,enc2,dec3_2]))

	g_bn11 = batch_norm(self.batch_size,name='g_bn11')
	dec4 = tf.image.resize_nearest_neighbor(dec3_3,[128,128])
	dec4 = tf.pad(dec4,[[0,0],[1,1],[1,1],[0,0]])
	dec4_1 = g_bn11(conv2d(dec4,self.df_dim*2,k_h=3,k_w=3,padding='VALID',name='g_dec4_1'),train=False) #output 128x128
	dec4_2 = tf.nn.tanh(conv2d(dec4_1,3,k_h=3,k_w=3,padding='SAME',name='g_dec4_2'))
	dec4_3 = lrelu(tf.concat(3,[dec4_1,enc1,dec4_2]))

	g_bn12 = batch_norm(self.batch_size,name='g_bn12')
	dec5 = tf.image.resize_nearest_neighbor(dec4,[256,256])
	dec5 = tf.pad(dec5,[[0,0],[1,1],[1,1],[0,0]])
	dec5_1 = g_bn12(conv2d(dec5,self.df_dim,k_h=3,k_w=3,padding='VALID',name='g_dec5_1'),train=False) #output 256x256
	dec5_2 = lrelu(tf.concat(3,[dec5_1,enc0]))
	dec6 = conv2d(dec5_2,3,k_h=1,k_w=1,name='g_dec6') #output 256x256

	return tf.nn.tanh(dec6)
