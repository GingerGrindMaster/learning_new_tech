import pika, socket, argparse, threading

""" CAST 2 - TOPIC EXCHANGE"""

# Funkce pro posílání zpráv
def send_event_te(message, routing_key):
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    # Vytvoření Topic Exchange
    channel.exchange_declare(exchange='topic_logs', exchange_type='topic')

    # Odeslání zprávy
    channel.basic_publish(exchange='topic_logs', routing_key=routing_key, body=message)
    print(f"Sent event: '{message}' with routing key '{routing_key}'")

    connection.close()


def read_events_te(routing_key, queue_name):
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    channel.exchange_declare(exchange='topic_logs', exchange_type='topic')
    channel.queue_declare(queue=queue_name)

    # Přímo připojíme frontu k danému routing klíči
    channel.queue_bind(exchange='topic_logs', queue=queue_name, routing_key=routing_key)

    def callback(ch, method, properties, body):
        print(f"[Listener] Received: {body.decode()} with key {method.routing_key}")

    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
    print(f"[Listener] Waiting for messages on {queue_name}. To exit press CTRL+C")
    channel.start_consuming()


if __name__ == '__main__':

    # Nastavení listenerů
    parser = argparse.ArgumentParser(description="RabbitMQ Event App")
    parser.add_argument('--send1', action='store_true', help="Send an event")
    parser.add_argument('--send2', action='store_true', help="Send an event")
    parser.add_argument('--send3', action='store_true', help="Send an event")
    parser.add_argument('--send4', action='store_true', help="Send an event")
    parser.add_argument('--read', action='store_true', help="Read events")
    args = parser.parse_args()

    # Pokud čteme zprávy
    if args.read:
        listeners = [
            threading.Thread(target=read_events_te, args=("user.*", "user_queue")),  # poslouchá na 'user.*'
            threading.Thread(target=read_events_te, args=("order.#", "topic_queue")),  # poslouchá na 'topic.x'
            threading.Thread(target=read_events_te, args=("order.#", "scnd_queue")),  # poslouchá na 'topic.x'
            threading.Thread(target=read_events_te, args=("different_topic.key", "other_queue")),
            # poslouchá na 'different_topic.key'
        ]

        # Spustíme všechny listenery
        for listener in listeners:
            listener.start()

        # Čekáme na ukončení všech listenerů (nebo ukončení programu)
        for listener in listeners:
            listener.join()

    # Pokud posíláme zprávy
    elif args.send1:
        send_event_te("listener 1 User logged in", "user.info")
    elif args.send2:
        send_event_te("listener 2 Order #123 created", "order.created")
    elif args.send3:
        send_event_te("listener 2 Order #123 cancelled", "order.cancelled")
    elif args.send4:
            send_event_te("different", "othertopic.x")

    else:
        print("Please specify --send or --read")

# Run the app.py file in the terminals with the following command:
# python app2.py --read
# python app2.py --send1
# python app2.py --send2
# ...