import Config as C
from ConfigTree import Config
from LLM import ChatContext
import os
import logging

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", default=C.SRCTREE, type=str)
    parser.add_argument("-t", "--type", default=0, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-o", "--output", default="config_output", type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    C.DEBUG = args.debug

    # disable openai output
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)

    # set OS environment
    os.environ["srctree"] = args.path
    os.environ["CC"] = C.CC
    os.environ["LD"] = C.LD
    os.environ["ARCH"] = C.ARCH
    os.environ["SRCARCH"] = C.SRCARCH

    # init llm chatter
    opt_id = args.type
    chatter = ChatContext(C.opt_target[opt_id], C.opt_description[opt_id], C.KEY)

    # read config and process
    config = Config(f"{args.path}/Kconfig", chatter, config_path=f"{args.path}/.config")
    config.run()
    config.save(args.output)
    print("Money spent on LLM: ", chatter.price)


if __name__ == "__main__":
    main()
