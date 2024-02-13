import sys

def hello(name="World"):
    msg = f"Hello, {name}!"
    print("*"*(len(msg)+4))
    print("*", msg, "*")
    print("*"*(len(msg)+4))
    return msg

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "World"
    hello(name)