from textblob import TextBlob

text = "five stars".replace('.','')
blob = TextBlob (text)
# 分句
print ("blob对象")
print (blob)
print (blob.sentiment)
