import re
from pypinyin import lazy_pinyin, Style
import json
import os

def generate_stock_list():
    # 从txt文件读取股票数据
    with open('china_stocks_20241231_154358.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()[2:]  # 跳过前两行标题
        
    stocks = []
    for line in lines:
        if line.strip() and not line.startswith('-'):  # 跳过分隔线
            parts = line.strip().split('\t')
            if len(parts) == 4:
                code, symbol, name, market = parts
                # 生成拼音（全拼和首字母）
                pinyin_full = ''.join(lazy_pinyin(name))
                pinyin_first = ''.join(lazy_pinyin(name, style=Style.FIRST_LETTER))
                
                baostock_code = f"{'sh' if market == 'SH' else 'sz'}.{code}"
                stock = {
                    'code': code,
                    'symbol': symbol,
                    'name': name,
                    'market': market,
                    'baostock_code': baostock_code,
                    'pinyin_full': pinyin_full,
                    'pinyin_first': pinyin_first
                }
                stocks.append(stock)
    
    # 确保static/js目录存在
    os.makedirs('static/js', exist_ok=True)
    
    # 生成JavaScript代码
    js_content = """// 股票列表数据
window.stockList = %s;

// 模糊匹配函数
window.fuzzyMatch = function(text, query) {
    text = text.toLowerCase();
    query = query.toLowerCase();
    
    let i = 0;
    let j = 0;
    let matches = false;
    
    while (i < text.length) {
        if (text[i] === query[j]) {
            j++;
            if (j === query.length) {
                matches = true;
                break;
            }
        }
        i++;
    }
    
    return matches;
}

// 添加搜索函数
window.searchStocks = function(query) {
    if (!query || query.length === 0) return [];
    
    return window.stockList.filter(stock => {
        // 完全匹配优先
        if (stock.code.toLowerCase().includes(query.toLowerCase()) ||
            stock.name.toLowerCase().includes(query.toLowerCase())) {
            return true;
        }
        
        // 拼音完全匹配其次
        if (stock.pinyin_full.toLowerCase().includes(query.toLowerCase()) ||
            stock.pinyin_first.toLowerCase().includes(query.toLowerCase())) {
            return true;
        }
        
        // 最后尝试模糊匹配
        return window.fuzzyMatch(stock.code, query) ||
               window.fuzzyMatch(stock.name, query) ||
               window.fuzzyMatch(stock.pinyin_full, query) ||
               window.fuzzyMatch(stock.pinyin_first, query);
    }).slice(0, 100); // 限制返回结果数量
}
""" % json.dumps(stocks, ensure_ascii=False, indent=2)
    
    # 保存到文件
    with open('static/js/stockList.js', 'w', encoding='utf-8') as f:
        f.write(js_content)
    
    print(f"Successfully generated stockList.js with {len(stocks)} stocks")

if __name__ == '__main__':
    generate_stock_list() 