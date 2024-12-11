import pika, socket, argparse

""" CAST 1 """

def send_event():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))  # pripojeni na RabbitMQ server bezici na localhostu
    channel = connection.channel()  # kanal je prostrednik pro komunikaci s RabbitMQ serverem

    channel.queue_declare(queue='simple_queue')  # deklarace fronty, pokud neexistuje, vytvori se

    msg = "Hello World!"
    channel.basic_publish(exchange='', routing_key='q', body = msg)

    # Získání hostname
    hostname = socket.gethostname()
    print(f"[{hostname}] Sent: {msg}")
    connection.close()

def on_message(ch, method, properties, body):
    print(f"Received: {body.decode()}")
    ch.basic_ack(delivery_tag=method.delivery_tag)  # Potvrzení zprávy


def read_event():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='q')
    channel.basic_consume(queue='q', on_message_callback=on_message, auto_ack=False)
    print('Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RabbitMQ Event App")
    parser.add_argument('--send', action='store_true', help="Send an event")
    parser.add_argument('--read', action='store_true', help="Read events")
    args = parser.parse_args()


    if args.send:
        send_event()
    elif args.read:
        read_event()
    else:
        print("Please specify --send or --read")





