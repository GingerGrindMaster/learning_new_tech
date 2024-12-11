import pika, argparse, threading
from datetime import datetime


def send_event_fanout(message):
    # Připojení k RabbitMQ
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    # Deklarace fanout exchange
    channel.exchange_declare(exchange='fanout_exchange', exchange_type='fanout')

    # Publikování zprávy do exchange
    channel.basic_publish(exchange='fanout_exchange', routing_key='', body=message)
    print(f"Posláno: {message}")

    connection.close()


def read_events_fanout(queue_name, process_message):
    """Poslouchá zprávy na frontě připojené k fanout exchange a zpracovává je pomocí zadané funkce."""
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    # Deklarace fanout exchange
    channel.exchange_declare(exchange='fanout_exchange', exchange_type='fanout')

    # Vytvoření a připojení fronty k exchange
    channel.queue_declare(queue=queue_name)
    channel.queue_bind(exchange='fanout_exchange', queue=queue_name)

    # Callback při obdržení zprávy
    def callback(ch, method, properties, body):
        processed_message = process_message(body.decode())
        print(f"[{queue_name}] {processed_message}")

    # Přihlášení k odběru zpráv z fronty
    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
    print(f"[{queue_name}] Čekám na zprávy...")

    channel.start_consuming()


if __name__ == "__main__":
    # Funkce pro různé typy zpracování zpráv
    def process_message_1(msg):
        return f"Listener 1 přijal zprávu: {msg.upper()}"


    def process_message_2(msg):
        return f"Listener 2 přeložil zprávu: {msg.replace('Hello', 'Ahoj')}"


    def process_message_3(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Listener 3 přijal zprávu {msg} v {timestamp}"


    # Argumenty příkazové řádky
    parser = argparse.ArgumentParser(description="RabbitMQ Fanout Exchange Demo")
    parser.add_argument('--send', action='store_true', help="Send an event")
    parser.add_argument('--read', action='store_true', help="Read events")
    parser.add_argument('--message', type=str, help="Message to send (for --send)")
    args = parser.parse_args()

    if args.read:
        # Seznam front a odpovídajících zpracovacích funkcí
        listeners = [
            ('fanout_queue_1', process_message_1),
            ('fanout_queue_2', process_message_2),
            ('fanout_queue_3', process_message_3),
        ]

        # Inicializace vlákna pro každý listener
        threads = []
        for queue_name, process_func in listeners:
            thread = threading.Thread(target=read_events_fanout, args=(queue_name, process_func))
            threads.append(thread)
            thread.start()

        # Čekání na dokončení všech vláken
        for thread in threads:
            thread.join()

    elif args.send:
        if not args.message:
            print("Pro režim --send musíte zadat zprávu pomocí --message.")
        else:
            # Poslání zprávy
            send_event_fanout(args.message)

    else:
        print("Musíte zadat buď --send nebo --read. Použijte --help pro více informací.")

# Run the following commands in separate terminals:
#python app3.py --send --message "ahoj"
#python app3.py --read
