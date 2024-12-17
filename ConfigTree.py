import kconfiglib as klib
import LLM
import logging


class Config:
    def __init__(
        self, kconfig_path: str, chatter: LLM.ChatContext, config_path: str = ".config"
    ):
        self.kconfig = klib.Kconfig(kconfig_path)
        self.kconfig.load_config(config_path)
        self.chatter = chatter
        self.current_node: klib.MenuNode = self.kconfig.top_node
        self.unvisit_node_list: list[klib.MenuNode] = [self.kconfig.top_node]
        self.node_dir_dict: dict[klib.MenuNode, list[str]] = {
            self.kconfig.top_node: [self.kconfig.top_node.prompt[0]]
        }
        logging.basicConfig(
            level=logging.INFO,
            filename="Config.log",
            datefmt="%Y/%m/%d %H:%M:%S",
        )

        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.FileHandler("Config.log", mode="w"))
        self.logger.propagate = False

    def run(self):
        while len(self.unvisit_node_list) > 0:
            self.current_node = self.unvisit_node_list.pop()
            print(f"Visiting menu {"/".join(self.node_dir_dict[self.current_node])}")
            self.process()

    def process(self):
        # get extended node list
        nodes = self.get_menunodes(self.current_node)

        menu_nodes = []
        bool_nodes = []
        binary_nodes = []
        trinary_nodes = []
        multiple_nodes = []
        value_nodes = []

        # iterate all current level nodes
        for node in nodes:
            item = node.item  # determine node type through this property
            if item == klib.MENU:
                menu_nodes.append(node)
            elif item == klib.COMMENT:
                # TODO: ignore comment currently
                pass
            else:
                # symbol or choice node
                if item.type in (klib.STRING, klib.INT, klib.HEX):
                    value_nodes.append(node)
                # select visible choice node
                elif (
                    isinstance(item, klib.Choice)
                    and item.visibility == 2
                    and item.str_value == "y"
                ):
                    multiple_nodes.append(node)
                elif len(item.assignable) == 1 and node.list:
                    # this node is a menu and is set to 'y' always
                    menu_nodes.append(node)
                elif item.type == klib.BOOL:
                    bool_nodes.append(node)
                elif item.type == klib.TRISTATE:
                    if item.assignable == (1, 2):
                        binary_nodes.append(node)
                    else:
                        trinary_nodes.append(node)

        # process all nodes
        if len(multiple_nodes) != 0:
            self.process_multiple(multiple_nodes)
        if len(value_nodes) != 0:
            self.process_value(value_nodes)
        if len(binary_nodes) != 0:
            self.process_binary(binary_nodes)
        if len(trinary_nodes) != 0:
            self.process_trinary(trinary_nodes)
        # bool nodes may be a menu. add these bool nodes to menu nodes if they are enabled
        new_menu_nodes = []
        if len(bool_nodes) != 0:
            new_menu_nodes.extend(self.process_bool(bool_nodes))

        # add menu nodes to unvisited nodes(which would be explored)
        menu_nodes.extend(new_menu_nodes)
        if len(menu_nodes) != 0:
            self.unvisit_node_list.extend(self.extend_nodes(menu_nodes))

    def get_menunodes(self, node: klib.MenuNode) -> list[klib.MenuNode]:
        """this function is used to get all active child nodes of a menunode

        Returns:
            list[klib.MenuNode]: child node list
        """
        node: klib.MenuNode = node.list
        # get all menu nodes to ask ai which nodes should be extended
        node_list = []
        while node:
            if node.prompt:
                if klib.expr_value(node.prompt[1]):
                    item = node.item
                    if isinstance(item, klib.Symbol) or isinstance(item, klib.Choice):
                        if item.type != klib.UNKNOWN:
                            node_list.append(node)
                    else:
                        node_list.append(node)
            node = node.next
        return node_list

    def extend_nodes(self, nodes: list[klib.MenuNode]) -> list[klib.MenuNode]:
        """
        this function is used to get menunodes to be extended
        for example, if a menu has 10 sub-nodes, ask llm to know which menunodes should be extended
        """
        node_name_list = []
        for i in range(len(nodes)):
            node_name_list.append(f"{i} {self.get_node_name(nodes[i])}")
        # ask LLM
        content = "\n".join(node_name_list)
        answers = self.chatter.ask_menu(
            content=content,
            mode=1 if self.current_node != self.kconfig.top_node else 0,
        )
        # answers is in form of [(0, General setup), (5, Kernel features), (6, Boot options), ...]
        menu_node: list[klib.MenuNode] = []
        # get node path prefix
        path = self.node_dir_dict[self.current_node]
        try:
            for answer in answers:
                node = nodes[answer[0]]
                menu_node.append(node)
                self.node_dir_dict[node] = path + [node.prompt[0]]
        except IndexError:
            error_answer = "\n".join([f"{t[0]} {t[1]}" for t in answers])
            self.logger.error(
                f"LLM gives non-existent nodes. current node is\n{content}\nLLM gives\n{error_answer}"
            )
        print("content:\n", content)
        print("selection:\n", "\n".join([f"{t[0]} {t[1]}" for t in answers]))
        return menu_node

    def process_bool(self, nodes: list[klib.MenuNode]) -> list[klib.MenuNode]:
        new_menu_nodes_dict: dict[str, klib.MenuNode] = {}
        # ask at most 15 config for once
        nodes_group = []
        for i in range(0, len(nodes), 15):
            nodes_group.append(nodes[i : i + 15])
        for group in nodes_group:
            node_name_dict = {}
            node_name_lower_dict = {}
            for node in group:
                name = self.get_node_name(node)
                node_name_dict[name] = node
                node_name_lower_dict[name.lower()] = node
                # add new menu nodes if node is enabled and node has child
                if node.item.tri_value == 2 and node.list:
                    new_menu_nodes_dict[name.lower()] = node
            node_names = "\n".join(node_name_dict.keys())
            # answer is a dict[str: int]
            answer = self.chatter.ask_on_off_option(node_names)
            for config_name, state in answer.items():
                config_name = config_name.lower()
                if config_name in node_name_lower_dict.keys():
                    node = node_name_lower_dict[config_name]
                    if node.item.tri_value == state:
                        continue
                    # log
                    self.logger.info(
                        f"Config changed: {node.item.name} from state '{node.item.tri_value}' to '{state}'"
                    )
                    # set config value to on or off. state is an int in (0, 2), where off = 0 and on = 2
                    node.item.set_value(state)
                    # if an option is set, check if it is a menu. if so, add it to menu node list
                    if state == 2:
                        new_menu_nodes_dict[config_name] = node
                    elif state == 0 and config_name in new_menu_nodes_dict.keys():
                        new_menu_nodes_dict.pop(config_name)
        return new_menu_nodes_dict.values()

    def process_binary(self, nodes: list[klib.MenuNode]):
        pass

    def process_trinary(self, nodes: list[klib.MenuNode]):
        pass

    def process_multiple(self, nodes: list[klib.MenuNode]):
        nodes_choices: list[list[klib.MenuNode]] = []
        choices_name_to_node_dict_list: list[dict[str, klib.MenuNode]] = []
        question_str_list = []
        for node in nodes:
            # get name of choices
            choices: list[klib.MenuNode] = []
            choices_name_to_node_dict: dict[str, klib.MenuNode] = {}
            node_list = self.get_menunodes(node)
            for choice in node_list:
                choices.append(choice)
                name = self.get_node_name(choice)
                question_str_list.append(name)
                choices_name_to_node_dict[name.lower()] = choice
            question_str_list.append("///")
            nodes_choices.append(choices)
            choices_name_to_node_dict_list.append(choices_name_to_node_dict)
        # get answers from LLM
        answers = self.chatter.ask_multiple_option("\n".join(question_str_list[:-1]))
        # answer is a list of choice names, each name indicates which config should be chosen
        assert len(answers) == len(nodes_choices)
        for i in range(len(answers)):
            answer = answers[i].lower()
            choices_name_to_node_dict = choices_name_to_node_dict_list[i]
            if answer in choices_name_to_node_dict.keys():
                node = choices_name_to_node_dict[answer]
                if node.item.tri_value == 2:
                    continue
                # log
                self.logger.info(
                    f"Config changed: {node.item.name} from state 'n' to 'y'"
                )
                choices_name_to_node_dict[answer].item.set_value(
                    2
                )  # enable this choice

    def process_value(self, nodes: list[klib.MenuNode]):
        # strings passed to LLM is in form of
        # stack depot hash size (12 => 4KB, 20 => 1024KB) (STACK_HASH_ORDER) (20)\n
        # etc.
        help_info_list = []
        node_info_list = []
        prompt_to_node_dict = {}

        # this code is from menuconfig.py
        def get_help_info_from_sym(sym: klib.Symbol):
            tristate_name = ["n", "m", "y"]
            prompt = f"Value for {sym.name}"
            if sym.type in (klib.BOOL, klib.TRISTATE):
                prompt += f" (available: {", ".join(tristate_name[val] for val in sym.assignable)})"
            prompt += ":"
            return f"{str(sym)}\n{prompt}"

        for node in nodes:
            item = node.item
            help_info_list.append(get_help_info_from_sym(item))
            node_info_list.append(f"{node.prompt[0]} ({item.str_value})")
            prompt_to_node_dict[node.prompt[0]] = node

        # call LLM
        answers = self.chatter.ask_value_option(
            "\n".join(help_info_list), "\n".join(node_info_list)
        )
        # answers is a list of tuple, where tuple[0] is prompt and tuple[1] is value
        # postprocess: set the value
        for answer in answers:
            if answer[0] in prompt_to_node_dict.keys():
                node = prompt_to_node_dict[answer[0]]
                if node.item.str_value == answer[1]:
                    continue
                # log
                self.logger.info(
                    f"Config changed: {node.item.name} from state '{node.item.str_value}' to '{answer[1]}'"
                )
                prompt_to_node_dict[answer[0]].item.set_value(answer[1])

    def save(self, path: str):
        self.kconfig.write_config(path)

    def get_node_name(self, node: klib.MenuNode):
        name = node.prompt[0]
        item = node.item
        if hasattr(item, "name"):
            name = f"{name} ({item.name})"
        return name
