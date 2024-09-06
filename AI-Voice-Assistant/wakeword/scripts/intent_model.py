import tensorflow as tf
from transformers import TFBertModel, BertTokenizer


class IntentModel():

    def __init__(self):
        self.MAX_LENGTH = 32
        self.classes = ['Question', 'Search', 'New Calendar', 'Read Calendar', 'Send Emails', 'Read Emails', 'Call contact', 'new contact', 'weather', 'open app', 'Read notification', 'Translation', 'Rejection', 'Acceptance', 'greetings', 'alarm']
        self.tokenizer =  BertTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02-twitter")

        with tf.device('/GPU:0'):  
            self.model = tf.keras.models.load_model('./intent model/intent_model.h5', custom_objects={'TFBertModel': TFBertModel})
    
    
    def get_intent(self, txt):
            ids = self.tokenizer(txt, return_tensors="tf", padding='max_length', max_length=self.MAX_LENGTH, truncation=True)['input_ids']
            
            with tf.device('/GPU:0'):
                pred = self.model.predict(ids)
            
            index = tf.math.argmax(pred[0])

            return self.classes[index]

# m = IntentModel()
# print(m.get_intent("الجو عامل ايه النهردة"))