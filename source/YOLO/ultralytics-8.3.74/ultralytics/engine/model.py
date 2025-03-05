    def device(self) -> torch.device:
        """
        Retrieves the device on which the model's parameters are allocated.
        # 获取模型参数所在的设备。
        This property determines the device (CPU or GPU) where the model's parameters are currently stored. It is
        applicable only to models that are instances of torch.nn.Module.
        # 该属性确定模型参数当前存储的设备（CPU或GPU）。仅适用于torch.nn.Module的实例。
        Returns:
            (torch.device): The device (CPU/GPU) of the model.
            # 返回模型的设备（CPU/GPU）。
        Raises:
            AttributeError: If the model is not a torch.nn.Module instance.
            # 如果模型不是torch.nn.Module的实例，则引发AttributeError。
        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.device)
            device(type='cuda', index=0)  # if CUDA is available
            >>> model = model.to("cpu")
            >>> print(model.device)
            device(type='cpu')
        """
        # 获取模型参数所在的设备
        return next(self.model.parameters()).device if isinstance(self.model, torch.nn.Module) else None  

    @property
    def transforms(self):
        """
        Retrieves the transformations applied to the input data of the loaded model.
        # 获取加载模型输入数据的变换。
        This property returns the transformations if they are defined in the model. The transforms
        typically include preprocessing steps like resizing, normalization, and data augmentation
        that are applied to input data before it is fed into the model.
        # 如果模型中定义了变换，则返回变换。变换通常包括在输入数据送入模型之前应用的预处理步骤，如调整大小、归一化和数据增强。
        Returns:
            (object | None): The transform object of the model if available, otherwise None.
            # 如果可用，返回模型的变换对象，否则返回None。
        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> transforms = model.transforms
            >>> if transforms:
            ...     print(f"Model transforms: {transforms}")
            ... else:
            ...     print("No transforms defined for this model.")
        """
        # 如果模型有'transforms'属性，则返回，否则返回None。
        return self.model.transforms if hasattr(self.model, "transforms") else None  

    def add_callback(self, event: str, func) -> None:
        """
        Adds a callback function for a specified event.
        # 为指定事件添加回调函数。
        This method allows registering custom callback functions that are triggered on specific events during
        model operations such as training or inference. Callbacks provide a way to extend and customize the
        behavior of the model at various stages of its lifecycle.
        # 此方法允许注册自定义回调函数，这些函数在模型操作（如训练或推理）期间的特定事件上被触发。回调提供了一种在模型生命周期的各个阶段扩展和自定义模型行为的方法。
        Args:
            event (str): The name of the event to attach the callback to. Must be a valid event name recognized
                by the Ultralytics framework.
            # event（str）：要将回调附加到的事件名称。必须是Ultralytics框架识别的有效事件名称。
            func (Callable): The callback function to be registered. This function will be called when the
                specified event occurs.
            # func（Callable）：要注册的回调函数。当指定事件发生时，将调用此函数。
        Raises:
            ValueError: If the event name is not recognized or is invalid.
            # 如果事件名称未被识别或无效，则引发ValueError。
        Examples:
            >>> def on_train_start(trainer):
            ...     print("Training is starting!")
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", on_train_start)
            >>> model.train(data="coco8.yaml", epochs=1)
        """
        # 将回调函数添加到指定事件的回调列表中。
        self.callbacks[event].append(func)  

    def clear_callback(self, event: str) -> None:
        """
        Clears all callback functions registered for a specified event.
        # 清除为指定事件注册的所有回调函数。
        This method removes all custom and default callback functions associated with the given event.
        # 此方法移除与给定事件相关的所有自定义和默认回调函数。
        It resets the callback list for the specified event to an empty list, effectively removing all
        registered callbacks for that event.
        # 将指定事件的回调列表重置为空列表，从而有效地移除所有已注册的回调。
        Args:
            event (str): The name of the event for which to clear the callbacks. This should be a valid event name
                recognized by the Ultralytics callback system.
            # event（str）：要清除回调的事件名称。这应该是Ultralytics回调系统识别的有效事件名称。
        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", lambda: print("Training started"))
            >>> model.clear_callback("on_train_start")
            >>> # All callbacks for 'on_train_start' are now removed
        """
        # 将指定事件的回调列表重置为空列表。
        self.callbacks[event] = []  

    def reset_callbacks(self) -> None:
        """
        Resets all callbacks to their default functions.
        # 将所有回调重置为默认函数。
        This method reinstates the default callback functions for all events, removing any custom callbacks that were
        previously added. It iterates through all default callback events and replaces the current callbacks with the
        default ones.
        # 此方法恢复所有事件的默认回调函数，移除之前添加的任何自定义回调。它遍历所有默认回调事件，并用默认回调替换当前回调。
        The default callbacks are defined in the 'callbacks.default_callbacks' dictionary, which contains predefined
        functions for various events in the model's lifecycle, such as on_train_start, on_epoch_end, etc.
        # 默认回调在'callbacks.default_callbacks'字典中定义，该字典包含模型生命周期中各种事件的预定义函数，例如on_train_start、on_epoch_end等。
        This method is useful when you want to revert to the original set of callbacks after making custom
        modifications, ensuring consistent behavior across different runs or experiments.
        # 当你想在进行自定义修改后恢复到原始回调集时，此方法非常有用，以确保在不同的运行或实验中保持一致的行为。
        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", custom_function)
            >>> model.reset_callbacks()
            # All callbacks are now reset to their default functions
        """
        # 将指定事件的回调列表重置为默认回调。
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]  

    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:
        """
        Resets specific arguments when loading a PyTorch model checkpoint.
        # 加载PyTorch模型检查点时重置特定参数。
        This static method filters the input arguments dictionary to retain only a specific set of keys that are
        considered important for model loading. It's used to ensure that only relevant arguments are preserved
        when loading a model from a checkpoint, discarding any unnecessary or potentially conflicting settings.
        # 此静态方法过滤输入参数字典，以仅保留被认为对模型加载重要的一组特定键。它用于确保在从检查点加载模型时，仅保留相关参数，丢弃任何不必要或可能冲突的设置。
        Args:
            args (dict): A dictionary containing various model arguments and settings.
            # args（dict）：包含各种模型参数和设置的字典。
        Returns:
            (dict): A new dictionary containing only the specified include keys from the input arguments.
            # （dict）：一个新字典，仅包含输入参数中指定的包含键。
        Examples:
            >>> original_args = {"imgsz": 640, "data": "coco.yaml", "task": "detect", "batch": 16, "epochs": 100}
            >>> reset_args = Model._reset_ckpt_args(original_args)
            >>> print(reset_args)
            {'imgsz': 640, 'data': 'coco.yaml', 'task': 'detect'}
        """
        # 返回仅包含指定键的字典。
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}  

    # def __getattr__(self, attr):
    #    """Raises error if object has no requested attribute."""
    #    name = self.__class__.__name__
    #    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.
    #    {self.__doc__}")

    def _smart_load(self, key: str):
        """
        Loads the appropriate module based on the model task.
        # 根据模型任务加载适当的模块。
        This method dynamically selects and returns the correct module (model, trainer, validator, or predictor)
        based on the current task of the model and the provided key. It uses the task_map attribute to determine
        the correct module to load.
        # 此方法根据模型的当前任务和提供的键动态选择并返回正确的模块（模型、训练器、验证器或预测器）。它使用task_map属性来确定要加载的正确模块。
        Args:
            key (str): The type of module to load. Must be one of 'model', 'trainer', 'validator', or 'predictor'.
            # key（str）：要加载的模块类型。必须是'model'、'trainer'、'validator'或'predictor'之一。
        Returns:
            (object): The loaded module corresponding to the specified key and current task.
            # （object）：与指定键和当前任务对应的加载模块。
        Raises:
            NotImplementedError: If the specified key is not supported for the current task.
            # 如果指定的键不支持当前任务，则引发NotImplementedError。
        Examples:
            >>> model = Model(task="detect")
            >>> predictor = model._smart_load("predictor")
            >>> trainer = model._smart_load("trainer")
        """
        # 根据模型任务返回对应的模块。
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(
                emojis(f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")
            ) from e

    @property
    def task_map(self) -> dict:
        """
        Provides a mapping from model tasks to corresponding classes for different modes.
        # 提供从模型任务到不同模式的对应类的映射。
        This property method returns a dictionary that maps each supported task (e.g., detect, segment, classify)
        to a nested dictionary. The nested dictionary contains mappings for different operational modes
        (model, trainer, validator, predictor) to their respective class implementations.
        # 此属性方法返回一个字典，将每个支持的任务（例如，detect、segment、classify）映射到嵌套字典。嵌套字典包含不同操作模式（模型、训练器、验证器、预测器）到其各自类实现的映射。
        The mapping allows for dynamic loading of appropriate classes based on the model's task and the
        desired operational mode. This facilitates a flexible and extensible architecture for handling
        various tasks and modes within the Ultralytics framework.
        # 该映射允许根据模型的任务和所需的操作模式动态加载适当的类。这为在Ultralytics框架中处理各种任务和模式提供了灵活和可扩展的架构。
        Returns:
            (Dict[str, Dict[str, Any]]): A dictionary where keys are task names (str) and values are
            nested dictionaries. Each nested dictionary has keys 'model', 'trainer', 'validator', and
            'predictor', mapping to their respective class implementations.
            # （Dict[str, Dict[str, Any]]）：一个字典，其中键是任务名称（str），值是嵌套字典。每个嵌套字典具有'模型'、'训练器'、'验证器'和'预测器'的键，映射到各自的类实现。
        Examples:
            >>> model = Model()
            >>> task_map = model.task_map
            >>> detect_class_map = task_map["detect"]
            >>> segment_class_map = task_map["segment"]
        
        Note:
            The actual implementation of this method may vary depending on the specific tasks and
            classes supported by the Ultralytics framework. The docstring provides a general
            description of the expected behavior and structure.
            # 此方法的实际实现可能因Ultralytics框架支持的特定任务和类而异。文档字符串提供了预期行为和结构的一般描述。
        """
        # 引发NotImplementedError，提示提供模型的任务映射。
        raise NotImplementedError("Please provide task map for your model!")  

    def eval(self):
        """
        Sets the model to evaluation mode.
        # 将模型设置为评估模式。
        This method changes the model's mode to evaluation, which affects layers like dropout and batch normalization
        that behave differently during training and evaluation.
        # 此方法将模型的模式更改为评估模式，这会影响在训练和评估期间表现不同的层，如dropout和batch normalization。
        Returns:
            (Model): The model instance with evaluation mode set.
            # （Model）：设置评估模式的模型实例。
        Examples:
            >> model = YOLO("yolo11n.pt")
            >> model.eval()
        """
        # 调用模型的eval方法，并返回模型实例。
        self.model.eval()
        return self  

    def __getattr__(self, name):
        """
        Enables accessing model attributes directly through the Model class.
        # 允许通过Model类直接访问模型属性。
        This method provides a way to access attributes of the underlying model directly through the Model class
        instance. It first checks if the requested attribute is 'model', in which case it returns the model from
        the module dictionary. Otherwise, it delegates the attribute lookup to the underlying model.
        # 此方法提供了一种通过Model类实例直接访问底层模型属性的方法。它首先检查请求的属性是否为'model'，如果是，则从模块字典中返回模型。否则，它将属性查找委托给底层模型。
        Args:
            name (str): The name of the attribute to retrieve.
            # name（str）：要检索的属性名称。
        Returns:
            (Any): The requested attribute value.
            # （Any）：请求的属性值。
        Raises:
            AttributeError: If the requested attribute does not exist in the model.
            # 如果请求的属性在模型中不存在，则引发AttributeError。
        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.stride)
            >>> print(model.task)
        """
        # 如果请求的属性是'model'，则返回模型；否则返回底层模型的属性。
        return self._modules["model"] if name == "model" else getattr(self.model, name) 