

# Tensorflow Implementation of conv net
version: r0.12

## Note on restoring pre-trained model

1. When calling saver.save(sess,path), 3 file will generated: 
    > checkpoint    # this file is a text file, indicating last saved model
    > model.ckpt.data-00000-of-00001   # this file is your save model name
    > model.ckpt.index  # 
    > model.ckpt.meta   #
2. There is no model file with specified name 
    Solution: Copy *model.ckpt.data-00000-of-00001* to *model.ckpt*, note to keep origin file

3. Use code block to restore
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, "checkpoint/model.ckpt")