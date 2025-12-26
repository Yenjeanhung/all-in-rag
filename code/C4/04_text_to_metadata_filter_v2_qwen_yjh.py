import os
from langchain_community.document_loaders import BiliBiliLoader
from langchain.chains.query_constructor.base import AttributeInfo
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging

""" 
这个是使用的公司的Qwen模型
"""

logging.basicConfig(level=logging.INFO)

# 1. 初始化视频数据
video_urls = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU", 
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
]

bili = []
try:
    loader = BiliBiliLoader(video_urls=video_urls)
    docs = loader.load()
    
    for doc in docs:
        original = doc.metadata
        
        # 提取基本元数据字段
        metadata = {
            'title': original.get('title', '未知标题'),
            'author': original.get('owner', {}).get('name', '未知作者'),
            'source': original.get('bvid', '未知ID'),
            'view_count': original.get('stat', {}).get('view', 0),
            'length': original.get('duration', 0),
        }
        
        doc.metadata = metadata
        bili.append(doc)
        
except Exception as e:
    print(f"加载BiliBili视频失败: {str(e)}")

if not bili:
    print("没有成功加载任何视频，程序退出")
    exit()

# 2. 创建向量存储
# embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
embed_model = HuggingFaceEmbeddings(
    model_name="E:/server/model/huggingface_cache/hub/models--BAAI--bge-small-zh-v1.5/snapshots/7999e1d3359715c523056ef9478215996d62a620"
)
vectorstore = Chroma.from_documents(bili, embed_model)

# 3. 配置元数据字段信息
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="视频标题（字符串）",
        type="string", 
    ),
    AttributeInfo(
        name="author",
        description="视频作者（字符串）",
        type="string",
    ),
    AttributeInfo(
        name="view_count",
        description="视频观看次数（整数）",
        type="integer",
    ),
    AttributeInfo(
        name="length",
        description="视频长度（整数）",
        type="integer"
    )
]

# 4. 初始化LLM客户端
client = OpenAI(
    base_url="http://192.168.20.68:3000/v1",
    api_key="sk-f11Yuhk0kymwzVBy15Ba991b77A343B684744c4d775e0d95"
)

# 5. 获取所有文档用于排序
all_documents = vectorstore.similarity_search("", k=len(bili)) 

# 6. 执行查询示例
queries = [
    "时间最短的视频",
    "播放量最高的视频"
]

for query in queries:
    print(f"/n--- 原始查询: '{query}' ---")

    # 使用大模型将自然语言转换为排序指令
    prompt = f"""你是一个智能助手，请将用户的问题转换成一个用于排序视频的JSON指令。

你需要识别用户想要排序的字段和排序方向。
- 排序字段必须是 'view_count' (观看次数) 或 'length' (时长) 之一。
- 排序方向必须是 'asc' (升序) 或 'desc' (降序) 之一。

例如:
- '时间最短的视频' 或 '哪个视频时间最短' 应转换为 {{"sort_by": "length", "order": "asc"}}
- '播放量最高的视频' 或 '哪个视频最火' 应转换为 {{"sort_by": "view_count", "order": "desc"}}

请根据以下问题生成JSON指令:
原始问题: "{query}"

JSON指令:"""
    
    response = client.chat.completions.create(
        model="QwQ-32B",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_tokens=15000
    )
    
    try:
        import json
        """ 
        TODO Qwen返回的内容中带有think标签，需要去掉,只保留json内容
        instruction_str：'<think>\n好的，用户的问题是要找出时间最短的视频。首先我需要确定用户想要排序的字段和方向。根据问题中的关键词“时间最短”，这里的“时间”应该指的是视频的时长，也就是length字段。而“最短”意味着升序排序，也就是从短到长排列。\n\n接下来，我需要检查是否符合要求的字段和方向。排序字段必须是view_count或length，这里明显是length。排序方向是asc还是desc，这里用户想要最短的，所以是asc。然后按照例子中的格式，生成对应的JSON指令。确认没有其他可能的字段或方向被误用，比如用户没有提到观看次数，所以不需要考虑view_count。最终确定JSON应该是{"sort_by": "length", "order": "asc"}。\n</think>\n\n```json\n{"sort_by": "length", "order": "asc"}\n```'
        """
        instruction_str = response.choices[0].message.content
        instruction = json.loads(instruction_str)
        print(f"--- 生成的排序指令: {instruction} ---")

        sort_by = instruction.get('sort_by')
        order = instruction.get('order')

        if sort_by in ['length', 'view_count'] and order in ['asc', 'desc']:
            # 在代码中执行排序
            reverse_order = (order == 'desc')
            sorted_docs = sorted(all_documents, key=lambda doc: doc.metadata.get(sort_by, 0), reverse=reverse_order)
            
            # 获取排序后的第一个结果
            if sorted_docs:
                doc = sorted_docs[0]
                title = doc.metadata.get('title', '未知标题')
                author = doc.metadata.get('author', '未知作者')
                view_count = doc.metadata.get('view_count', '未知')
                length = doc.metadata.get('length', '未知')
                print(f"标题: {title}")
                print(f"作者: {author}")
                print(f"观看次数: {view_count}")
                print(f"时长: {length}秒")
                print("="*50)
            else:
                print("没有找到任何视频")
        else:
            print("生成的指令无效，无法执行排序")

    except (json.JSONDecodeError, KeyError) as e:
        print(f"解析或执行指令失败: {e}")
