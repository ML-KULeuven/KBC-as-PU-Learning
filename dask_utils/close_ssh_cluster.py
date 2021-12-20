from distributed import Client


def main():
    scheduler_host = 'pinac31'

    # client = Client(cluster)
    # scheduler_address="tcp://134.58.41.100:8786"
    scheduler_address = f'{scheduler_host}:8786'
    client = Client(address=scheduler_address)
    client.shutdown()
    print("Shutting down client, its connected scheduler and workers...")


if __name__ == '__main__':
    main()