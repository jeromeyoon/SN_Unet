from ops import *
import tensorflow as tf

class networks(object):
    def __init__(self,num_block,batch_size,df_dim,keep_prob):
	self.num_block = num_block
	self.batch_size = batch_size
	self.df_dim = df_dim
	self.keep_prob = keep_prob 
    def generator(self,nir):
	#### encoder ####
        enc0 = lrelu(conv2d(nir,self.df_dim,name='g_enc0'))
	g_bn1 = batch_norm(self.batch_size,name='g_bn1')
        enc1 =lrelu(g_bn1(conv2d(enc0,self.df*2,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc1'))) #output:128x128
	g_bn2 = batch_norm(self.batch_size,name='g_bn2')
        enc2 =lrelu(g_bn2(conv2d(enc1,self.df*4,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc2'))) #output:64x64
	g_bn3 = batch_norm(self.batch_size,name='g_bn3')
        enc3 =lrelu(g_bn3(conv2d(enc2,self.df*8,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc3'))) #output:32x32
	g_bn4 = batch_norm(self.batch_size,name='g_bn4') 
        enc4 =lrelu(g_bn4(conv2d(enc3,self.df*8,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc4'))) #output:16x16
	g_bn5 = batch_norm(self.batch_size,name='g_bn5')
        enc5 =lrelu(g_bn5(conv2d(enc4,self.df*8,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc5'))) #output:8x8
	g_bn6 = batch_norm(self.batch_size,name='g_bn6')
        enc6 =lrelu(g_bn6(conv2d(enc5,self.df*8,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc6'))) #output:4x4
	g_bn7 = batch_norm(self.batch_size,name='g_bn7')
        enc7 =tf.nn.relu(g_bn6(conv2d(enc6,self.df*8,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc7'))) #output:2x2
	#### Deconder #####
	g_bn8 = batch_norm(self.batch_size,name='g_bn8')
	dec1 = tf.nn.dropout(g_bn8(deconv2d(enc7,[self.batch_size,self.df*8,4,4],,name='g_dec1')),self.keep_prob)#output 4x4
	dec1 = tf.nn.relu(tf.concat(3,[dec1,enc6]))
	g_bn9 = batch_norm(self.batch_size,name='g_bn9')
	dec2 = tf.nn.dropout(g_bn9(deconv2d(dec1,[self.batch_size,self.df*8,8,8],,name='g_dec2')),self.keep_prob)#output 8x8
	dec2 = tf.nn.relu(tf.concat(3,[dec2,enc5]))
	g_bn10 = batch_norm(self.batch_size,name='g_bn10')
	dec3 = tf.nn.dropout(g_bn10(deconv2d(dec2,[self.batch_size,self.df*8,16,16],,name='g_dec3')),self.keep_prob)#output 16x16
	dec3 = tf.nn.relu(tf.concat(3,[dec3,enc4]))
	g_bn11 = batch_norm(self.batch_size,name='g_bn11')
	dec4 = g_bn11(deconv2d(dec3,[self.batch_size,self.df*8,32,32],,name='g_dec4'))#output 32x32
	dec4 = tf.nn.relu(tf.concat(3,[dec4,enc3]))
	g_bn12 = batch_norm(self.batch_size,name='g_bn12')
	dec5 = g_bn12(deconv2d(dec4,[self.batch_size,self.df*4,64,64],,name='g_dec5'))#output 64x64
	dec5 = tf.nn.relu(tf.concat(3,[dec5,enc2]))
	g_bn13 = batch_norm(self.batch_size,name='g_bn13')
	dec6 = g_bn13(deconv2d(dec5,[self.batch_size,self.df*2,128,128],,name='g_dec6'))#output 128x128
	dec6 = tf.nn.relu(tf.concat(3,[dec6,enc1]))
	dec7 = deconv2d(dec6,[self.batch_size,3,256,256],,name='g_dec7')#output 256x256
	return tf.nn.tanh(dec7)

    def discriminator(self, image, reuse=False):
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
        h4 = d_bn4(conv2d(h3,1,k_w=4,k_h=4,d_h=1,d_w=1,name='d_h4_conv')) #output size 16x 16
        h5 = conv2d(h4,1,k_w=4,k_h=4,d_h=1,d_w=1,name='d_h5_conv') #output size 16x16
        return tf.nn.sigmoid(h5)

    def sampler(self,nir):
	#### encoder ####
        enc0 = lrelu(conv2d(nir,self.df_dim,name='g_enc0'))
	g_bn1 = batch_norm(self.batch_size,name='g_bn1')
        enc1 =lrelu(g_bn1(conv2d(enc0,self.df*2,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc1'),train=False)) #output:128x128
	g_bn2 = batch_norm(self.batch_size,name='g_bn2')
        enc2 =lrelu(g_bn2(conv2d(enc1,self.df*4,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc2'),train=False)) #output:64x64
	g_bn3 = batch_norm(self.batch_size,name='g_bn3')
        enc3 =lrelu(g_bn3(conv2d(enc2,self.df*8,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc3'),train=False)) #output:32x32
	g_bn4 = batch_norm(self.batch_size,name='g_bn4') 
        enc4 =lrelu(g_bn4(conv2d(enc3,self.df*8,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc4'),train=False)) #output:16x16
	g_bn5 = batch_norm(self.batch_size,name='g_bn5')
        enc5 =lrelu(g_bn5(conv2d(enc4,self.df*8,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc5'),train=False)) #output:8x8
	g_bn6 = batch_norm(self.batch_size,name='g_bn6')
        enc6 =lrelu(g_bn6(conv2d(enc5,self.df*8,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc6'),train=False)) #output:4x4
	g_bn7 = batch_norm(self.batch_size,name='g_bn7')
        enc7 =tf.nn.relu(g_bn6(conv2d(enc6,self.df*8,k_h=4,k_w=4,d_h=2,d_w=2,name='g_enc7'),train=False)) #output:2x2
	#### Deconder #####
	g_bn8 = batch_norm(self.batch_size,name='g_bn8')
	dec1 = g_bn8(deconv2d(enc7,[self.batch_size,self.df*8,4,4],,name='g_dec1'),train=False)#output 4x4
	dec1 = tf.nn.relu(tf.concat(3,[dec1,enc6]))
	g_bn9 = batch_norm(self.batch_size,name='g_bn9')
	dec2 = g_bn9(deconv2d(dec1,[self.batch_size,self.df*8,8,8],name='g_dec2'),train=False)#output 8x8
	dec2 = tf.nn.relu(tf.concat(3,[dec2,enc5]))
	g_bn10 = batch_norm(self.batch_size,name='g_bn10')
	dec3 = g_bn10(deconv2d(dec2,[self.batch_size,self.df*8,16,16],name='g_dec3'),train=False)#output 16x16
	dec3 = tf.nn.relu(tf.concat(3,[dec3,enc4]))
	g_bn11 = batch_norm(self.batch_size,name='g_bn11')
	dec4 = g_bn11(deconv2d(dec3,[self.batch_size,self.df*8,32,32],name='g_dec4'),train=False)#output 32x32
	dec4 = tf.nn.relu(tf.concat(3,[dec4,enc3]))
	g_bn12 = batch_norm(self.batch_size,name='g_bn12')
	dec5 = g_bn12(deconv2d(dec4,[self.batch_size,self.df*4,64,64],name='g_dec5'),train=False)#output 64x64
	dec5 = tf.nn.relu(tf.concat(3,[dec5,enc2]))
	g_bn13 = batch_norm(self.batch_size,name='g_bn13')
	dec6 = g_bn13(deconv2d(dec5,[self.batch_size,self.df*2,128,128],name='g_dec6'),train=False)#output 128x128
	dec6 = tf.nn.relu(tf.concat(3,[dec6,enc1]))
	dec7 = deconv2d(dec6,[self.batch_size,3,256,256],,name='g_dec7')#output 256x256

	return tf.nn.tanh(dec7)
