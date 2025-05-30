#!/usr/bin/env python3
"""
Weather MCP Usage Examples - 天气查询MCP服务使用示例
展示如何在不同场景下使用天气MCP服务
"""

import asyncio
import json
from test_weather_mcp import WeatherMCPClient

async def example_basic_weather_query():
    """示例1: 基本天气查询"""
    print("=" * 50)
    print("示例1: 基本天气查询")
    print("=" * 50)
    
    client = WeatherMCPClient()
    
    try:
        await client.start_server()
        await client.initialize()
        
        # 查询多个城市的天气
        cities = ["北京", "上海", "广州", "深圳"]
        
        for city in cities:
            print(f"\n🔍 查询 {city} 的天气...")
            await client.get_weather(city)
            
    finally:
        await client.close()

async def example_weather_forecast():
    """示例2: 天气预报查询"""
    print("\n" + "=" * 50)
    print("示例2: 天气预报查询")
    print("=" * 50)
    
    client = WeatherMCPClient()
    
    try:
        await client.start_server()
        await client.initialize()
        
        # 查询不同天数的天气预报
        city = "杭州"
        for days in [1, 3, 5]:
            print(f"\n🔍 查询 {city} 未来{days}天的天气预报...")
            await client.get_weather_forecast(city, days)
            
    finally:
        await client.close()

async def example_batch_weather_query():
    """示例3: 批量天气查询"""
    print("\n" + "=" * 50)
    print("示例3: 批量天气查询")
    print("=" * 50)
    
    client = WeatherMCPClient()
    
    try:
        await client.start_server()
        await client.initialize()
        
        # 批量查询多个城市
        cities_with_countries = [
            ("北京", "CN"),
            ("Tokyo", "JP"),
            ("New York", "US"),
            ("London", "GB"),
            ("Paris", "FR")
        ]
        
        print("🌍 全球城市天气查询:")
        for city, country in cities_with_countries:
            print(f"\n🔍 查询 {city}, {country} 的天气...")
            
            # 发送自定义请求
            response = await client.send_request("tools/call", {
                "name": "get_weather",
                "arguments": {
                    "city": city,
                    "country_code": country
                }
            })
            
            if "result" in response:
                content = response["result"]["content"][0]["text"]
                print(content)
            else:
                print(f"❌ 查询失败: {response}")
                
    finally:
        await client.close()

async def example_error_handling():
    """示例4: 错误处理"""
    print("\n" + "=" * 50)
    print("示例4: 错误处理演示")
    print("=" * 50)
    
    client = WeatherMCPClient()
    
    try:
        await client.start_server()
        await client.initialize()
        
        # 测试无效城市名称
        invalid_cities = ["", "不存在的城市", "123456"]
        
        for city in invalid_cities:
            print(f"\n🔍 测试无效城市: '{city}'")
            try:
                if city:  # 只有非空字符串才发送请求
                    await client.get_weather(city)
                else:
                    print("❌ 空城市名称，跳过查询")
            except Exception as e:
                print(f"❌ 查询出错: {e}")
                
    finally:
        await client.close()

async def example_custom_mcp_requests():
    """示例5: 自定义MCP请求"""
    print("\n" + "=" * 50)
    print("示例5: 自定义MCP请求")
    print("=" * 50)
    
    client = WeatherMCPClient()
    
    try:
        await client.start_server()
        await client.initialize()
        
        # 获取工具列表
        print("📋 获取可用工具列表:")
        tools_response = await client.send_request("tools/list")
        if "result" in tools_response:
            tools = tools_response["result"]["tools"]
            for i, tool in enumerate(tools, 1):
                print(f"  {i}. {tool['name']}: {tool['description']}")
        
        # 发送自定义天气查询
        print("\n🔧 发送自定义天气查询请求:")
        custom_request = {
            "name": "get_weather",
            "arguments": {
                "city": "成都",
                "country_code": "CN"
            }
        }
        
        response = await client.send_request("tools/call", custom_request)
        if "result" in response:
            content = response["result"]["content"][0]["text"]
            print(content)
        
        # 测试天气预报
        print("\n🔧 发送自定义天气预报请求:")
        forecast_request = {
            "name": "get_weather_forecast",
            "arguments": {
                "city": "成都",
                "days": 4
            }
        }
        
        response = await client.send_request("tools/call", forecast_request)
        if "result" in response:
            content = response["result"]["content"][0]["text"]
            print(content)
            
    finally:
        await client.close()

async def main():
    """运行所有示例"""
    print("🌤️ 天气查询MCP服务使用示例")
    print("=" * 60)
    
    examples = [
        ("基本天气查询", example_basic_weather_query),
        ("天气预报查询", example_weather_forecast),
        ("批量天气查询", example_batch_weather_query),
        ("错误处理演示", example_error_handling),
        ("自定义MCP请求", example_custom_mcp_requests)
    ]
    
    for name, example_func in examples:
        try:
            print(f"\n🚀 运行示例: {name}")
            await example_func()
            await asyncio.sleep(1)  # 稍作延迟
        except Exception as e:
            print(f"❌ 示例 '{name}' 运行失败: {e}")
    
    print("\n✅ 所有示例运行完成!")

if __name__ == "__main__":
    asyncio.run(main())