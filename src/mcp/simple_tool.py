from datetime import datetime

from lunardate import LunarDate

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import TransportSecuritySettings
import random


def get_local_time_(detail=True):
    """获取当前时间。详细格式"""
    # 公历
    cur_time = datetime.now()
    cur_time = cur_time.timetuple()

    if detail:
        date_str = f"当前时间：{cur_time.tm_year}年{cur_time.tm_mon}月{cur_time.tm_mday}日，{cur_time.tm_hour}时{cur_time.tm_min}分{cur_time.tm_sec}秒，星期"
        date_str += ["一", "二", "三", "四", "五", "六", "日"][cur_time.tm_wday]
        date_str += ";"

        # 农历
        lunar = LunarDate.fromSolarDate(
            cur_time.tm_year, cur_time.tm_mon, cur_time.tm_mday
        )
        date_str += f" 农历：{lunar.year}年 {lunar.month}月 {lunar.day}日"
        date_str += f" 是否闰月：{'是' if lunar.isLeapMonth else '否'}。"

        return date_str
    else:
        date_str = (
            f"当前时间：{cur_time.tm_year}年{cur_time.tm_mon}月{cur_time.tm_mday}日"
        )
        return date_str


server = FastMCP(
    "simple_tool",
    stateless_http=True,
    json_response=True,
    streamable_http_path="/mcp",  # 改这里
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
        allowed_hosts=["*"],
        allowed_origins=["*"],
    ),
)


@server.tool()
def get_current_time() -> str:
    """获取当前时间: 详细的时间信息格式(包括公历和农历)"""
    return get_local_time_()


@server.tool()
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    # 这里可以集成一个真实的天气API，暂时返回模拟数据

    if random.random() < 0.1:
        return f"{city} 天气是阴天，可能会下雨，出门记得带伞！"
    else:
        return f"{city}的天气是晴朗，温度25摄氏度，湿度60%。"


@server.tool()
def add(a: float, b: float) -> str:
    """计算两个数的和"""
    return f"{a} + {b} = {a + b}"


@server.tool()
def multiply(a: float, b: float) -> str:
    """计算两个数的积"""
    return f"{a} * {b} = {a * b}"


@server.tool()
def devision(a: float, b: float) -> str:
    """计算两个数的商"""
    if b == 0:
        return "除数不能为零！"
    return f"{a} / {b} = {a / b}"


@server.tool()
def minus(a: float, b: float) -> str:
    """计算两个数的差"""
    return f"{a} - {b} = {a - b}"
