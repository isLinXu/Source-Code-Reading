'''
@Description: 
@Author: jiajunlong
@Date: 2024-06-19 19:30:17
@LastEditTime: 2024-06-19 19:32:47
@LastEditors: jiajunlong
'''
class Message:
    # 初始化Message类，可以传入初始消息列表，如果没有则默认为空列表
    def __init__(self, msg=None):
        self._messages = msg if msg else []
        self._images = []                        # 初始化图片列表为空
        self.skip_next = False                   # 初始化跳过下一条消息的标志为False

    # 添加消息的方法，接受问题和答案（可选）
    def add_message(self, question, answer=None):
        quension_msg_dict = {'from': 'human'}    # 创建表示问题的字典，来源为'human'
        quension_msg_dict['value'] = question    # 将问题添加到字典中
        answer_msg_dict = {'from': 'gpt'}        # 创建表示答案的字典，来源为'gpt'
        answer_msg_dict['value'] = answer        # 将答案添加到字典中
        self._messages.append(quension_msg_dict) # 将问题字典添加到消息列表
        self._messages.append(answer_msg_dict)   # 将答案字典添加到消息列表

    # 添加图片的方法，可以指定插入图片的索引位置，默认为0
    def add_image(self, image, index=0):
        self._images.append((image, index))      # 将图片和索引位置作为元组添加到图片列表

    # 获取图片列表的只读属性
    @property
    def images(self):
        return self._images    

    # 获取消息列表的只读属性
    @property
    def messages(self):
        return self._messages

    # 复制Message实例的方法，返回一个新的Message实例，包含相同的消息列表
    def copy(self):
        return Message(self._messages)

    # 将消息转换为Gradio聊天机器人格式的方法
    def to_gradio_chatbot(self):
        ret = []
        for i, msg in enumerate(self.messages):
            if i % 2 == 0:                                           # 如果是偶数索引，表示这是一条来自'human'的消息
                if len(self.images) != 0 and i == self.images[0][1]: # 如果有图片，并且当前消息索引与图片列表中的第一个图片索引相同
                    image = self.images[0][0]
                    import base64
                    from io import BytesIO
                    msg = msg['value']
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])                         # 将处理后的消息添加到返回列表
                else:
                    ret.append([msg['value'], None])                # 将消息值添加到返回列表
            else:                                                   # 如果是奇数索引，表示这是一条来自'gpt'的消息
                ret[-1][-1] = msg['value']                          # 将答案值添加到上一条消息的答案位置
        return ret                                                  # 返回处理后的消息列表