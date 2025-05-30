#!/usr/bin/env python3
"""
Weather MCP Client - 天气查询MCP服务客户端测试
演示如何与天气MCP服务器进行交互
"""

import asyncio
import json
import subprocess
import sys
from typing import Dict, Any

class WeatherMCPClient:
    """天气MCP客户端"""
    
    def __init__(self):
        self.process = None
        self.request_id = 0
    
    async def start_server(self):
        """启动MCP服务器进程"""
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, "weather_mcp_server.py",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        print("✅ MCP服务器已启动")
    
    async def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """发送请求到MCP服务器"""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": str(self.request_id),
            "method": method
        }
        
        if params:
            request["params"] = params
        
        # 发送请求
        request_json = json.dumps(request, ensure_ascii=False) + "\n"
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()
        
        # 读取响应
        response_line = await self.process.stdout.readline()
        response = json.loads(response_line.decode().strip())
        
        return response
    
    async def initialize(self):
        """初始化MCP连接"""
        response = await self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "weather-test-client",
                "version": "1.0.0"
            }
        })
        
        if "result" in response:
            print("✅ MCP连接初始化成功")
            print(f"服务器信息: {response['result']['serverInfo']}")
        else:
            print(f"❌ 初始化失败: {response}")
    
    async def list_tools(self):
        """获取可用工具列表"""
        response = await self.send_request("tools/list")
        
        if "result" in response:
            tools = response["result"]["tools"]
            print(f"\n📋 可用工具 ({len(tools)}个):")
            for tool in tools:
                print(f"  🔧 {tool['name']}: {tool['description']}")
        else:
            print(f"❌ 获取工具列表失败: {response}")
        
        return response.get("result", {}).get("tools", [])
    
    async def get_weather(self, city: str, country_code: str = ""):
        """查询天气"""
        params = {
            "name": "get_weather",
            "arguments": {
                "city": city
            }
        }
        
        if country_code:
            params["arguments"]["country_code"] = country_code
        
        response = await self.send_request("tools/call", params)
        
        if "result" in response:
            content = response["result"]["content"][0]["text"]
            print(f"\n{content}")
        else:
            print(f"❌ 天气查询失败: {response}")
    
    async def get_weather_forecast(self, city: str, days: int = 3):
        """查询天气预报"""
        params = {
            "name": "get_weather_forecast",
            "arguments": {
                "city": city,
                "days": days
            }
        }
        
        response = await self.send_request("tools/call", params)
        
        if "result" in response:
            content = response["result"]["content"][0]["text"]
            print(f"\n{content}")
        else:
            print(f"❌ 天气预报查询失败: {response}")
    
    async def close(self):
        """关闭客户端连接"""
        if self.process:
            self.process.terminate()
            await self.process.wait()
            print("✅ MCP服务器已关闭")

async def interactive_demo():
    """交互式演示"""
    client = WeatherMCPClient()
    
    try:
        # 启动服务器
        await client.start_server()
        
        # 初始化连接
        await client.initialize()
        
        # 获取工具列表
        await client.list_tools()
        
        print("\n" + "="*50)
        print("🌤️  天气查询MCP服务演示")
        print("="*50)
        
        while True:
            print("\n请选择操作:")
            print("1. 查询当前天气")
            print("2. 查询天气预报")
            print("3. 退出")
            
            choice = input("\n请输入选项 (1-3): ").strip()
            
            if choice == "1":
                city = input("请输入城市名称: ").strip()
                if city:
                    await client.get_weather(city)
                    
            elif choice == "2":
                city = input("请输入城市名称: ").strip()
                if city:
                    try:
                        days = int(input("请输入预报天数 (1-5): ").strip() or "3")
                        days = max(1, min(5, days))  # 限制在1-5天范围内
                        await client.get_weather_forecast(city, days)
                    except ValueError:
                        print("❌ 请输入有效的天数")
                        
            elif choice == "3":
                break
            else:
                print("❌ 无效选项，请重新选择")
    
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，正在退出...")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
    finally:
        await client.close()

async def automated_demo():
    """自动化演示"""
    client = WeatherMCPClient()
    
    try:
        print("🚀 启动自动化演示...")
        
        # 启动服务器
        await client.start_server()
        
        # 初始化连接
        await client.initialize()
        
        # 获取工具列表
        await client.list_tools()
        
        print("\n" + "="*50)
        print("🌤️  自动化天气查询演示")
        print("="*50)
        
        # 演示城市列表
        cities = ["北京", "上海", "广州", "深圳", "杭州"]
        
        for city in cities:
            print(f"\n🔍 正在查询 {city} 的天气...")
            await client.get_weather(city)
            await asyncio.sleep(1)  # 稍作延迟
        
        # 演示天气预报
        print(f"\n🔍 正在查询 {cities[0]} 的3天天气预报...")
        await client.get_weather_forecast(cities[0], 3)
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
    finally:
        await client.close()

async def main():
    """主函数"""
    print("🌤️ 天气查询MCP服务测试客户端")
    print("="*40)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        await automated_demo()
    else:
        await interactive_demo()

if __name__ == "__main__":
    asyncio.run(main())