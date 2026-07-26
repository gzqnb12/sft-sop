"""共享任务定义。

将 schema 集中在一个位置，避免训练和评测静默使用不同的标签空间。
"""

INTENTS = ("refund", "logistics", "account", "invoice", "product")
URGENCIES = ("low", "medium", "high")

SYSTEM_PROMPT = """你是客服工单分类器。读取客户消息，只输出一行 JSON，不要解释，不要 Markdown。
JSON 必须恰好包含两个字段：
- intent: refund、logistics、account、invoice、product 之一
- urgency: low、medium、high 之一

判定规则：
- refund：退款、退货退款、重复扣款
- logistics：发货、快递、包裹、配送
- account：登录、密码、账号安全、身份验证
- invoice：发票、抬头、税号、开票
- product：商品功能、质量、损坏、使用问题
- high：资金或账号安全风险，或明确要求当天解决
- medium：已有问题需要数日内处理
- low：一般咨询，没有明确时限或损失"""
