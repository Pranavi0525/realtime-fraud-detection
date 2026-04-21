from kafka import KafkaConsumer, TopicPartition

consumer = KafkaConsumer(bootstrap_servers='localhost:9092')
tp = TopicPartition('fraud-alerts', 0)
consumer.assign([tp])
consumer.seek_to_beginning(tp)

for message in consumer:
    print(message.value)