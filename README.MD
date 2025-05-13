# Chat Loop with OpenAI & MCP

## English 🇬🇧

### Overview

This project demonstrates how to build a **streaming, tool‑enabled chat loop** that connects
OpenAI’s GPT‑4o (or GPT‑4o‑mini) with an **MCP (Multi‑Capability Platform)** server. The script
creates an asynchronous conversation where the assistant can call remote tools exposed by the
MCP, handle streaming responses, and display final answers once all tool calls are resolved.

### Features

* **Asynchronous**: Built on `asyncio`, `openai.AsyncOpenAI`, and an async MCP client.
* **Streaming completions**: See assistant responses as they generate.
* **Dynamic tool usage**: Discovers available MCP tools at runtime and lets the model invoke them.
* **Automatic argument assembly**: Collects partial tool‑call deltas before executing the tool.

### Folder Structure

```
.
├── chat_loop.py      # Main script (the code in this repository)
└── README.md         # You are here
```

### Prerequisites

| Requirement | Version        |
| ----------- | -------------- |
| Python      | 3.10 or higher |
| pip         | latest         |

Install dependencies:

```bash
pip install openai mcp
```

*(If your MCP client lives in a different package name, adjust accordingly.)*

### Starting a Local MCP Server (Playwright implementation)

If you don’t already have an MCP server running, you can spin up a quick local instance powered by
[Playwright MCP](https://www.npmjs.com/package/@playwright/mcp):

```bash
npx @playwright/mcp@latest --port 8931
```

This will start an MCP server at `http://localhost:8931` and print something like:

```text
Listening on http://localhost:8931
Put this in your client config:
{
  "mcpServers": {
    "playwright": {
      "url": "http://localhost:8931/sse"
    }
  }
}
If your client supports streamable HTTP, you can use the /mcp endpoint instead.
```

*For the Python script in this repo, we use the streamable‑HTTP endpoint (`/mcp`).*
If you change the port or host, remember to update `MCP_URL` in **chat\_loop.py**.

### Configuration

Set an **environment variable** for your OpenAI key or replace the placeholder string in the
script (not recommended for production):

```bash
export OPENAI_APIKEY="sk-...your-key..."
```

Ensure your MCP server is running locally at the default URL or set `MCP_URL`.

### Running the Script

```bash
python chat_loop.py "What's the weather like in Istanbul today?"
```

If no argument is provided, the script defaults to:

```text
What's the weather like in Istanbul today?
```

**Example interaction** (after starting the Playwright MCP server):

```bash
$ python chat_loop.py "İstanbul'da hava durumu nedir?"

💬 Assistant: İstanbul Havalimanı için hava durumu şu şekildedir:

- Durum: Çok Bulutlu
- Sıcaklık: 14,6 °C
- Basınç: 1016 hPa
- Rüzgar: 31 km/sa (Kuzeydoğudan)
- Nem: %63

Hava durumu 13 Mayıs 2025 tarihinde saat 12:50 için geçerlidir.
```

### How It Works (High Level)

1. **Tool Discovery** – `session.list_tools()` fetches tool metadata from the MCP server.
2. **OpenAI Chat Request** – The assistant receives the user prompt + tool definitions.
3. **Streaming Response** – As chunks arrive, the script collects normal text and any
   `tool_calls` emitted by the model.
4. **Tool Execution** – Once a complete tool‑call payload is assembled, the script invokes the
   corresponding MCP tool and appends the result to the chat history as a `tool` role message.
5. **Final Answer** – When the assistant eventually responds without further tool calls, the loop
   prints the answer and exits.

### Customising

* **Model Choice**: Change `model="gpt-4o-mini"` to any model your key has access to.
* **MCP Endpoint**: Edit `MCP_URL` if your MCP server is on a different host or port.
* **System / Assistant Prompts**: Prepend system instructions or modify the user prompt list as
  needed.

### License

MIT – feel free to adapt.

---

## Türkçe 🇹🇷

### Genel Bakış

Bu proje, OpenAI’nin GPT‑4o modelini **MCP (Multi‑Capability Platform)** sunucusuna bağlayarak
**akış (streaming) destekli ve araç kullanabilen** bir sohbet döngüsü kurmanın örneğini sunar.
Betik, asenkron olarak çalışan bir sohbet oluşturur; asistan MCP’nin sunduğu araçları
çağırabilir, akış hâlindeki yanıtları yönetir ve tüm araç çağrıları tamamlanınca nihai yanıtı
çıktılar.

### Özellikler

* **Asenkron** yapı (`asyncio`, `openai.AsyncOpenAI`, asenkron MCP istemcisi).
* **Streaming** tamamlamalar: Yanıtlar oluşturulurken anlık görebilirsiniz.
* **Dinamik araç kullanımı**: MCP’nin sağladığı araçlar çalışma zamanında keşfedilir ve model
  tarafından çağrılır.
* **Otomatik argüman birleştirme**: Parçalı `tool_call` deltalarını toplar ve tek seferde
  çalıştırır.

### Dosya Yapısı

```
.
├── chat_loop.py   # Ana betik
└── README.md      # Bu dosya
```

### Gereksinimler

| Gereksinim | Sürüm  |
| ---------- | ------ |
| Python     | 3.10+  |
| pip        | güncel |

Kurulum:

```bash
pip install openai mcp
```

### Yerel MCP Sunucusunu Başlatma (Playwright sürümü)

Henüz MCP sunucunuz yoksa, [Playwright MCP](https://www.npmjs.com/package/@playwright/mcp)
ile hızlıca yerel bir örnek başlatabilirsiniz:

```bash
npx @playwright/mcp@latest --port 8931
```

Bu komut `http://localhost:8931` adresinde bir MCP sunucusu başlatacak ve aşağıdakine benzer
çıktı üretecektir:

```text
Listening on http://localhost:8931
Put this in your client config:
{
  "mcpServers": {
    "playwright": {
      "url": "http://localhost:8931/sse"
    }
  }
}
If your client supports streamable HTTP, you can use the /mcp endpoint instead.
```

*Bu depodaki Python betiği `streamable‑HTTP` yolu (`/mcp`) kullanır.*
Bağlantı noktası veya host değiştirirseniz, **chat\_loop.py** içindeki `MCP_URL` sabitini
uygulayın.

### Yapılandırma

Aşağıdakilerden birini yapın:

1. Ortam değişkeni oluşturun:

   ```bash
   export OPENAI_APIKEY="sk-...anahtarınız..."
   ```
2. Veya **geçici** olarak betikteki `OPENAI_APIKEY` sabitini değiştirin *(üretim için tavsiye
   edilmez!)*.

`MCP_URL` değişkeni MCP sunucunuz farklıysa düzenlenmelidir.

### Çalıştırma

```bash
python chat_loop.py "İstanbul'da hava durumu nedir?"
```

Parametre verilmezse betik şu varsayılan soruyu kullanır:

```text
İstanbul'da hava durumu bugün nasıl?
```

**Örnek etkileşim** (Playwright MCP sunucusu başlatıldıktan sonra):

```bash
$ python chat_loop.py "İstanbul'da hava durumu nedir?"

💬 Assistant: İstanbul Havalimanı için hava durumu şu şekildedir:

- Durum: Çok Bulutlu
- Sıcaklık: 14,6 °C
- Basınç: 1016 hPa
- Rüzgar: 31 km/sa (Kuzeydoğudan)
- Nem: %63

Hava durumu 13 Mayıs 2025 tarihinde saat 12:50 için geçerlidir.
```

### Nasıl Çalışır (Özet)

1. **Araç Keşfi** – `session.list_tools()` MCP üzerindeki araçları alır.
2. **OpenAI Sohbet İsteği** – Asistan, kullanıcı önceden tanımlı araç listesiyle birlikte
   isteği alır.
3. **Akış Yanıtı** – Parçalar geldikçe normal metin ve `tool_calls` toplanır.
4. **Araç Çalıştırma** – Tam `tool_call` verisi oluştuğunda ilgili MCP aracı çağrılır ve sonucu
   `tool` rolünde sohbete eklenir.
5. **Final Yanıt** – Asistan yeni araç çağrısı olmadan yanıt verdiğinde döngü sonlanır.

### Özelleştirme

* **Model Seçimi**: `model="gpt-4o-mini"` değerini erişiminiz olan başka bir modele
  değiştirebilirsiniz.
* **MCP Adresi**: `MCP_URL`’yi sunucunuzun adresine göre ayarlayın.
* **Sistem/Asistan İletileri**: Gerekirse sistem talimatları ekleyin veya ileti geçmişini
  düzenleyin.

### Lisans

MIT – dilediğiniz gibi kullanın ve uyarlayın.
