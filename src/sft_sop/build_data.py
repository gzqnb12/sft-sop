"""Build a deterministic toy dataset for the SFT walkthrough.

The examples are synthetic and intentionally small. Different utterances and
prompt wrappers are reserved for each split so that the test set is not an
exact copy of training templates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sft_sop.constants import SYSTEM_PROMPT

CASE_BANK: dict[str, dict[str, dict[str, list[str]]]] = {
    "refund": {
        "high": {
            "train": [
                "平台刚刚连续扣了我两次款，请今天立刻退回多扣的钱",
                "退款页面显示成功但银行卡没有到账，今晚不到账我就投诉",
            ],
            "validation": ["同一订单被扣了三笔钱，请下班前把多余款项退回"],
            "test": ["系统误扣了我的生活费，今天必须原路退回"],
        },
        "medium": {
            "train": [
                "退货签收已经五天了，退款还是审核中",
                "订单取消三天了，我还没有收到退款",
            ],
            "validation": ["上周寄回的商品已经入库，麻烦尽快处理退款"],
            "test": ["退货完成四天后仍然没有退款进度"],
        },
        "low": {
            "train": [
                "想了解一下未发货订单的退款规则",
                "如果鞋码不合适，申请退款需要哪些材料",
            ],
            "validation": ["请问七天无理由退款是否需要承担运费"],
            "test": ["我想先咨询数字商品能不能申请退款"],
        },
    },
    "logistics": {
        "high": {
            "train": [
                "药品包裹显示丢失，病人今晚要用，请今天处理",
                "快递把贵重设备送错地址了，请立刻联系追回",
            ],
            "validation": ["生鲜一直没有配送，今天再不到就全部坏了"],
            "test": ["护照被快递送错城市，明早出发，必须马上找回"],
        },
        "medium": {
            "train": [
                "包裹三天没有更新物流信息，帮我查一下",
                "预计昨天送达的快递现在还在中转站",
            ],
            "validation": ["订单已经一周没发货，请核实仓库进度"],
            "test": ["快递连续两天显示派送中但没有联系我"],
        },
        "low": {
            "train": [
                "请问普通配送一般需要几天",
                "下单后可以把收货地址改到公司吗",
            ],
            "validation": ["周末快递是否正常发货"],
            "test": ["想知道你们合作的是哪家快递公司"],
        },
    },
    "account": {
        "high": {
            "train": [
                "账号出现陌生登录和订单，请立即冻结",
                "手机丢了且账号绑定着银行卡，请马上停用登录",
            ],
            "validation": ["有人修改了我的密码，请立刻帮我找回账号"],
            "test": ["账号正在异地消费，这不是我本人操作，请马上锁定"],
        },
        "medium": {
            "train": [
                "更换手机号后收不到验证码，已经两天无法登录",
                "实名认证一直失败，导致我不能提交订单",
            ],
            "validation": ["密码重置邮件等了一天还没有收到"],
            "test": ["账号被系统限制登录三天了，麻烦查明原因"],
        },
        "low": {
            "train": [
                "如何修改账号绑定的邮箱",
                "请问能不能设置两步验证",
            ],
            "validation": ["一个手机号最多可以注册几个账号"],
            "test": ["我想了解注销账号需要经过哪些步骤"],
        },
    },
    "invoice": {
        "high": {
            "train": [
                "项目今天验收必须提交发票，请立即补开",
                "发票税号开错导致财务无法付款，今天需要重开",
            ],
            "validation": ["报销今晚截止，请马上发送电子发票"],
            "test": ["客户今天结算，但发票抬头错误，请立即更正"],
        },
        "medium": {
            "train": [
                "申请开票三天了还没有收到电子发票",
                "纸质发票寄出后一直查不到物流",
            ],
            "validation": ["上周提交的增值税专票仍在审核中"],
            "test": ["公司发票缺少商品明细，需要这周内补充"],
        },
        "low": {
            "train": [
                "个人订单可以改开公司抬头吗",
                "电子发票在哪里下载",
            ],
            "validation": ["开具专票需要提供哪些资料"],
            "test": ["想咨询发票抬头是否可以包含英文名称"],
        },
    },
    "product": {
        "high": {
            "train": [
                "充电器使用时冒烟并有焦味，请今天处理安全问题",
                "儿童座椅卡扣突然断裂，存在安全风险，请立即联系",
            ],
            "validation": ["电池已经鼓包发热，请马上告诉我如何处置"],
            "test": ["热水壶漏电把人电到了，请立即安排处理"],
        },
        "medium": {
            "train": [
                "耳机到货后只有一边有声音，希望尽快更换",
                "新买的键盘用了两天就有多个按键失灵",
            ],
            "validation": ["显示器出现明显坏点，影响正常工作"],
            "test": ["空气净化器运行几分钟就自动关机"],
        },
        "low": {
            "train": [
                "这款相机支持多大的存储卡",
                "请问咖啡机第一次使用要如何清洗",
            ],
            "validation": ["这件外套的面料可以机洗吗"],
            "test": ["想了解手表是否支持游泳时佩戴"],
        },
    },
}

PROMPT_TEMPLATES = {
    "train": (
        "客户消息：{text}",
        "请为下面的客服消息分类：{text}",
    ),
    "validation": ("新工单内容如下，请判断分类：{text}",),
    "test": ("判断这条客户诉求的意图和紧急程度：{text}",),
}


def build_records() -> dict[str, list[dict[str, Any]]]:
    """Return train/validation/test records."""

    records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    counters = {split: 0 for split in records}

    for intent, urgency_groups in CASE_BANK.items():
        for urgency, split_groups in urgency_groups.items():
            answer = json.dumps(
                {"intent": intent, "urgency": urgency},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for split, texts in split_groups.items():
                for text in texts:
                    for template in PROMPT_TEMPLATES[split]:
                        counters[split] += 1
                        records[split].append(
                            {
                                "id": f"{split}-{counters[split]:03d}",
                                "chat_template_kwargs": {"enable_thinking": False},
                                "messages": [
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": template.format(text=text)},
                                    {"role": "assistant", "content": answer},
                                ],
                            }
                        )
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for train.jsonl, validation.jsonl and test.jsonl.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = build_records()
    for split, rows in records.items():
        path = args.output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        print(f"{split:>10}: {len(rows):>3} examples -> {path}")


if __name__ == "__main__":
    main()
