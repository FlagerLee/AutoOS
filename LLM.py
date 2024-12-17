from openai import OpenAI
import logging
import re

import Config as C


class ChatContext:
    def __init__(
        self,
        opt_target: str,
        opt_description: str,
        api_key: str,
        api_url: str = "https://api.ai.cs.ac.cn/v1",
        model: str = "gpt-3.5-turbo-1106",
    ):
        """
        You should choose an optimization target to start using LLM.
        Target is a string, for example, it could be "the Dhrystone and Whetstone scores in UnixBench", so that LLM
        can optimize config towards this goal.
        Then you should pass an optimization description to describe what the target does and how to improve it.
        """
        # init logging
        self.logger = logging.getLogger(__name__)

        # init LLM context
        self.client = OpenAI(base_url=api_url, api_key=api_key)
        self.model = model

        # init prompt set
        self.menu_prompt = [
            f"I want to explore the configuration of the Linux kernel's 'config' to {opt_description}. I will "
            f"sequentially show you each level of menuconfig's directories, and I need you to recommend  directories "
            f"related to target and filter any directory related to Soc selection, device driver , cryptographic,file "
            f"system, network ,platform type or architecture-depend directories. Here's how I'll show you the "
            f"directories which is in some level in menuconfig:\n  [number] [directory name] \nYour response format: \n"
            f"For relevant directories: [number] [directory name] \nFor example，when I give:0 memory setting(mem) \n"
            f"1 driver(d) \n   your answer :' 0 memory setting(mem) \n' No extra explanations needed. Do not recommend "
            f"any directory related to Soc selection, device driver , cryptographic,file system, network ,platform "
            f"type or architecture-depend directories. Do not mention reason.Please obey the rules. Here are some "
            f"directories,please  recommend:",
            f"I want to explore the configuration of the Linux kernel's 'config' to {opt_description}. I will "
            f"sequentially show you each level of menuconfig's directories, and I need you to tell me which "
            f"directories are possibly related to {opt_description}  based on your existing knowledge. Here's how I'll "
            f"show you the directories which is in some level in menuconfig:\n  [number] [directory name] \nYour "
            f"response format: \n For relevant directories: [number] [directory name] \nFor example，when I give:"
            f"0 memory setting(mem) \n  1 computer name(Name) \n   your answer :' 0 memory setting(mem) \n' because "
            f"the memory setting is related to {opt_description} but the name is not. \nNo extra explanations needed. "
            f"Do not recommend any directory related to Soc selection, device driver , cryptographic,file system, "
            f"network ,platform type or architecture-depend directories. Do not mention reason.Please obey the rules. "
            f"Here are some directories,please  recommend:",
        ]
        self.on_off_prompt = [
            f" For {opt_target}  , analyze each of the following settings separately to determine whether they will "
            f"increase or decrease {opt_target} if the setting is enabled:",
            f"Based on the analysis above, provide the options that could potentially affect {opt_target} and analyse "
            f"whether enabling the  settings will increase or decrease {opt_target}",
            f"According to the above analysis,for the options that could potentially impact {opt_target} , determine "
            f"whether each option will increase or decrease {opt_target}, Output format: 'increase: \n Option Name1 \n "
            f"Option Name2 \n  decrease: \n Option Name1 \n Option Name2'. No explation, no extra useless words. "
            f"For example，when related option is： IO Schedulers (IOS) \n  DYNAMIC_DEBUG(DD)   \n, the analysis is "
            f"that IO Schedulers (IOS)   will increase the score and  DYNAMIC_DEBUG(DD)   will decrease the score   "
            f"your answer : 'increase: \n IO Schedulers (IOS) \n  decrease: \n DYNAMIC_DEBUG(DD)  \n.Output complete "
            f"name ,for example,output 'IO Schedulers (IOS) ',do not output 'IOS ' only! Complete name is important,"
            f"you need attention. In the output, the assessment of enabling each option should align with the previous "
            f"analysis, indicating whether it will increase or decrease {opt_target}. Do not mention reason.Do not "
            f"output options about network or Peripheral drives, just ignore.The option names should maintain "
            f"consistent capitalization.  Please obey the rules. Output complete name as I said .please  output:",
        ]
        self.multiple_option_prompt = (
            f"I'm exploring the Linux kernel's menuconfig for configurations to "
            f"{opt_description}. Here are  multiple 'select one option' choices in "
            f"menuconfig. Please select one suitable option at a time to potentially "
            f"{opt_description}. My format:\n [option1 name] \n  [option2 name] \n  ..  \n"
            f"Your response format: \n '[recommended option name]\n' \nfor example:  when I "
            f"give: 'reveive buffer（rbuf）  \n log buffer(lbuf) \n /// \n  CPU schedule(cs) "
            f"\n  CPU  default(cd) \n /// \n SLAB (SLAB) \n SLUB (Unqueued Allocator) "
            f"(SLUB)'. This means there are three   'select one option' choices, and "
            f"considering to {opt_description} ,your answer is :receive buffer(rbuf)\n CPU "
            f"schedule(cs) \n SLUB (Unqueued Allocator) (SLUB)\nRemember to choose the "
            f"recommended setting to possibly {opt_description} for each option: No extra "
            f"explanations needed. Only suggest options which may be related. Do not "
            f"mention reason.Output complete name ,for example,output 'CPU schedule(cs)',do "
            f"not optput 'cs' only! Complete name is important,you need attention.Please "
            f"obey the rules. Here are some  'select one option' choices,please choose:"
        )
        self.binary_option_prompt = (
            f"I'm exploring the Linux kernel's menuconfig for configurations that might "
            f"{opt_description}. Here are  multiple binary choice options in menuconfig. "
            f"There are two settings for an option: 'M' or 'on', which ‘M’ means configuring "
            f"this option as a module and 'on' means compiling this option into the kernel "
            f"image to make it a part of the kernel. .Please set the  options at a time to "
            f"potentially {opt_description}. My format:\n [option name]   \n  Your response "
            f"format: \n [option name]  {{M or on}}\nfor example:  when I give: 'reveive "
            f"buffer(rbuf) \n log buffer(lbuf)\n ',your answer is 'receive buffer(rbuf) "
            f"{{on}} \n log buffer(lbuf) {{M}}' \nRemember to  recommend settings to possibly "
            f"{opt_description} for each option: No extra explanations needed. Only suggest "
            f"options which may be related. Do not mention reason. Please obey the rules. "
            f"Here are some binary choice options ,please  recommend:"
        )
        self.trinary_option_prompt = (
            f"I'm exploring the Linux kernel's menuconfig for configurations that might "
            f"{opt_description}. Here are  multiple ternary choice options in menuconfig. "
            f"There are three settings for an option: 'M' ，'on' or 'off', which ‘M’ means "
            f"configuring this option as a module , 'on' means compiling this option into "
            f"the kernel image to make it a part of the kernel. and 'off' means disabling "
            f"this option, not compiling it as a kernel component.Please set the  options at "
            f"a time to potentially {opt_description}. My format:\n [option name]   \n  Your "
            f"response format: \n [option name]  <M or on or  off>\nfor example:  when I "
            f"give: 'reveive buffer(rbuf)  \n log buffer(lbuf) \n Debug Filesystem(DFile) ',"
            f"your answer is 'receive buffer(rbuf) <on> \n  log buffer(lbuf) <M> \n Debug "
            f"Filesystem(DFile) <off>'   \nRemember to  recommend settings to possibly "
            f"{opt_description} for each option: No extra explanations needed. Only suggest "
            f"options which may be tie to {opt_description}. Do not mention reason. Please "
            f"obey the rules. Here are some ternary choice options ,please  recommend:"
        )
        self.value_option_prompt = (
            f"I'm exploring the Linux kernel's menuconfig for configurations that might "
            f"{opt_description}. Here are  multiple numeric  options in menuconfig.I have "
            f"given you the range of each option value in the information above. Please set "
            f"the  options at a time to potentially {opt_description}. If the option is not "
            f"rellated to {opt_description}, then remain the defalut value. \n My format:\n  "
            f"[option name] (default value)  \n  Your response format: \n [option name] "
            f"(recommended  value)   \nfor example:  when I give: 'maximum CPU number"
            f"(1=>2 2=>4)  (cpunum) (1)',your answer is 'maximum CPU number(1=>2 2=>4)  "
            f"(cpunum) (2)'   . Because  when the CPU number is more, the speed is usually "
            f"better.\nRemember to  recommend settings to possibly {opt_description} for each "
            f"option: No extra explanations needed. Only suggest options which maybe "
            f"{opt_description}. Do not mention reason. Do not add units near number in the "
            f"output.Please obey the rules. Here are some numeric options ,please  recommend: "
        )

        # init regex pattern used to extract answers from the answers
        self.menu_ans_pattern = re.compile(
            r"^(\d+)\s+(\[\s+\]|\[\*\]|\(.*?\)|\<.*?\>|\({.*?}\)|\-\*\-|\-\-\>)?\s*([A-Za-z].*?)\s*($|\n)"
        )
        self.value_ans_pattern = re.compile(
            r"(.*?)\s+([\[\(\<\{]?(?:on|off|M|\d+|\-\-\>)[\]\)\}\>]?)\s*($|\n)"
        )

        # init price accumulator
        self.price = 0.0

        # init logger
        logging.basicConfig(
            level=logging.INFO,
            datefmt="%Y/%m/%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.FileHandler("LLM.log", mode="w"))
        self.logger.propagate = False

    def chat(self, content: list[dict]) -> str:
        # log
        self.logger.info(f"LLM REQUEST:\n{content}")
        response = self.client.chat.completions.create(
            messages=content, model=self.model
        )
        # add prices
        self.price += (
            int(response.usage.prompt_tokens) * self.get_prompt_price()
            + int(response.usage.completion_tokens) * self.get_completion_price()
        )
        # log
        self.logger.info(f"LLM RESPONSE:\n{response.choices[0].message.content}")
        return response.choices[0].message.content

    def ask_menu(self, content: str, mode: int) -> list[tuple[int, str]]:
        conversation = [
            {"role": "user", "content": f"{self.menu_prompt[mode]} {content}"}
        ]
        answer = self.chat(conversation)
        answers = answer.split("\n")
        matches = []
        for ans in answers:
            m = self.menu_ans_pattern.findall(ans)
            if len(m) == 0:
                continue
            m = m[0]
            matches.append((int(m[0]), m[2]))
        return matches

    def ask_on_off_option(self, content: str) -> dict[str:int]:
        conversation = [
            {"role": "user", "content": f"{self.on_off_prompt[0]} {content}"}
        ]
        conversation.append({"role": "assistant", "content": self.chat(conversation)})
        conversation.append({"role": "user", "content": self.on_off_prompt[1]})
        conversation.append({"role": "assistant", "content": self.chat(conversation)})
        conversation.append({"role": "user", "content": self.on_off_prompt[2]})
        # answer is a concatenation of the following two forms
        # increase:\n xxx-config
        # decrease:\n xxx-config
        answer = self.chat(conversation)
        # extract options to be open and close
        result_dict = {}
        # 'lines' is a list, for example, lines = ['increase', 'xxx-config', 'decrease', 'xxx-config']
        lines = answer.split("\n")
        is_increase = False
        for line in lines:
            if line.lower() == "increase:":
                is_increase = True
                continue
            elif line.lower() == "decrease:":
                is_increase = False
                continue
            if line == "":
                continue
            result_dict[line] = 2 if is_increase else 0
        return result_dict

    def ask_multiple_option(self, content: str) -> list[str]:
        answer = self.chat(
            [{"role": "user", "content": f"{self.multiple_option_prompt} {content}"}]
        )
        return answer.split("\n")

    def ask_binary_option(self, content: str) -> str:
        return self.chat(
            [{"role": "user", "content": f"{self.binary_option_prompt} {content}"}]
        )

    def ask_trinary_option(self, content: str) -> str:
        return self.chat(
            [{"role": "user", "content": f"{self.trinary_option_prompt} {content}"}]
        )

    def ask_value_option(self, help_info: str, content: str) -> list[tuple[str, str]]:
        answer = self.chat(
            [
                {
                    "role": "user",
                    "content": f"Here is value options information: "
                    f"{help_info}\n{self.value_option_prompt} {content}",
                }
            ]
        )
        # get useful message from answer
        matches = self.value_ans_pattern.findall(answer)
        result = []
        for m in matches:
            # m is in form of ('Warn for stack frames larger than (FRAME_WARN)', '(512)', '')
            result.append((m[0], m[1][1:-1]))
        return result

    def get_prompt_price(self):
        return 0.008 / 1000

    def get_completion_price(self):
        return 0.016 / 1000
