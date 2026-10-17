// Incremental SSE parser. `feed` takes the next decoded chunk and returns the
// complete events it finished; a partial trailing frame is kept for the next
// chunk (a network read can split a frame anywhere, including mid-UTF8 line).
function createSSEParser() {
  let buffer = "";
  return {
    feed(chunk) {
      buffer += chunk;
      const events = [];
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        let type = "message";
        const data = [];
        for (const line of frame.split("\n")) {
          if (line.startsWith("event: ")) type = line.slice(7);
          else if (line.startsWith("data: ")) data.push(line.slice(6));
        }
        if (data.length) events.push({ type, data: JSON.parse(data.join("\n")) });
      }
      return events;
    },
  };
}
if (typeof module !== "undefined") module.exports = { createSSEParser };
