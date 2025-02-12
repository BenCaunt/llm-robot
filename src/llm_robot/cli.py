import click
import zenoh
import asyncio
from typing import Optional

@click.group()
def main():
    """LLM Robot CLI - Interact with the zenoh network"""
    pass

@main.command()
@click.option('--connect', '-c', default=None, help='Zenoh connect string')
def read(connect: Optional[str]):
    """Read all data from the zenoh network"""
    async def subscriber_callback(sample):
        click.echo(f"Received {sample.key_expr} = {sample.payload.decode('utf-8')}")

    async def main_task():
        conf = zenoh.Config()
        if connect:
            conf.insert_json5(f"{{connect: {connect}}}")
        
        session = await zenoh.open(conf)
        subscriber = await session.declare_subscriber("**", subscriber_callback)
        
        click.echo("Reading from zenoh network... Press Ctrl+C to stop")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await subscriber.undeclare()
            await session.close()

    asyncio.run(main_task())

if __name__ == '__main__':
    main() 