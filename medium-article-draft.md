# Single-Agent with Skills vs. Multi-Agent Systems: What the Research Shows

*Why a single agent with the right skills might be all you need—and when it isn't*

---

If you've been paying attention to the AI engineering space lately, you've probably noticed that "multi-agent" has become the default answer to almost every complex problem. Need to build a research assistant? Spin up a crew of agents. Want to automate a workflow? Deploy an agent swarm. Building something ambitious? Obviously, you need an orchestrator coordinating a team of specialized agents.

The multi-agent pattern has become so prevalent that reaching for it feels like the "serious" architectural choice. Single agents? That's for toy projects and demos.

But what if this complexity is often unnecessary? What if, for many use cases, we're paying a steep tax in tokens, latency, and debugging headaches—without getting proportional value in return?

A recent research paper challenges the multi-agent orthodoxy, and its findings deserve attention from anyone building agent-based systems.

---

## The Research: Single Agent with Skills vs. Multi-Agent Systems

The paper "When Single-Agent with Skills Replace Multi-Agent Systems and When They Fail" by Xiaoxiao Li investigates a deceptively simple question: Can a single agent with a curated library of skills replicate the benefits of multi-agent systems?

The answer, according to the research, is often yes—and more efficiently.

### The Core Finding

The researchers demonstrated that a single agent selecting from a skill library can substantially reduce token usage and latency while maintaining competitive accuracy on reasoning benchmarks. Instead of multiple agents passing messages, maintaining separate contexts, and coordinating handoffs, one agent simply picks the right skill for the task.

The key insight here is viewing skills as "internalized agent behaviors." Rather than spinning up a separate "research agent" and "writing agent" and "review agent," you teach one agent how to research, how to write, and how to review. The behaviors that would be distributed across multiple agents become skills within a single, unified context.

This isn't just a theoretical efficiency gain. In practice, multi-agent systems pay real costs:

- **Token overhead**: Each agent maintains its own context, often duplicating information
- **Latency**: Coordination and handoffs between agents take time
- **Complexity**: Debugging distributed agent failures is notoriously difficult
- **Unpredictability**: Inter-agent communication introduces new failure modes

A single agent with skills sidesteps most of these issues while preserving the capability to handle diverse tasks.

---

## The Phase Transition Problem: When It Breaks Down

Here's where the research gets really interesting—and provides crucial guidance for practitioners.

The single-agent approach works beautifully... until it suddenly doesn't.

### The Cliff, Not the Slope

You might expect that as you add more skills to an agent's library, performance would gradually degrade. A gentle slope downward as the agent gets slightly worse at picking the right skill from an increasingly crowded menu.

That's not what happens.

Instead, the research found a "phase transition"—skill selection accuracy remains stable up to a critical library size, then drops sharply. One moment the agent is reliably choosing appropriate skills; the next, it's confused and making poor selections. There's no graceful degradation, just a cliff.

This pattern mirrors something fascinating from cognitive science: the bounded capacity of human decision-making. Psychologists have long observed that humans can effectively juggle about seven (plus or minus two) distinct options before decision quality deteriorates. We don't get gradually worse at choosing between 10, 15, 20 options—we hit a wall where choice overload kicks in.

LLMs, it turns out, exhibit similar bounded capacity when selecting skills.

### The Real Culprit: Semantic Similarity

But here's the crucial nuance: the phase transition isn't purely about library size. The research identifies semantic similarity among skills as the primary driver of breakdown.

If your skill library contains "write_technical_doc," "write_documentation," "create_technical_content," and "author_doc_page"—congratulations, you've set a trap for your agent. These skills sound similar enough that the agent will struggle to distinguish between them, even in a relatively small library.

Conversely, a library with clearly distinct skills—"write_code," "run_tests," "search_web," "analyze_data"—can remain effective at larger sizes because the semantic distance between options is greater.

This finding has immediate practical implications: skill hygiene matters as much as skill count.

---

## Practical Implications: What This Means for You

So how should this research inform the way you build agent systems? Here are the key takeaways.

### Default to Single-Agent First

Before reaching for a multi-agent architecture, ask yourself: "Could one agent with the right skills handle this?" The research suggests that for many use cases, the answer is yes—with better efficiency.

Multi-agent systems should be a deliberate choice made after determining that single-agent won't work, not the default starting point for "serious" projects.

### Curate Skills with Distinct Purposes

If you're building a skill library, prioritize clarity and distinctiveness over comprehensiveness. Each skill should have a clearly differentiated purpose that an LLM can unambiguously identify.

Consider these questions when adding a new skill:
- Does this overlap semantically with existing skills?
- Could the agent reasonably confuse this with another skill?
- Is the skill name and description distinct enough to stand out?

Fewer, clearer skills will outperform a bloated library of overlapping capabilities.

### Watch for the Warning Signs

As your skill library grows, monitor for signs that you're approaching the phase transition:
- The agent starts picking "close but wrong" skills
- You find yourself adding elaborate prompting to help skill selection
- Accuracy on skill-dependent tasks begins to drop

These are signals that you've hit—or are approaching—the capacity limit.

### Consider Hierarchical Organization

The paper presents preliminary evidence that hierarchical routing can mitigate capacity limitations. Instead of presenting 50 skills to the agent at once, organize them into categories: "writing skills," "analysis skills," "code skills." The agent first selects a category, then selects a skill within that category.

This mirrors how humans manage complex choice environments—we chunk and categorize to reduce cognitive load.

### Know When Multi-Agent Still Makes Sense

None of this means multi-agent systems are never appropriate. They remain valuable for:

- **Truly parallel workloads**: When tasks genuinely can and should execute simultaneously
- **Specialized domain separation**: When different tasks require fundamentally different models or capabilities
- **Isolation requirements**: When you need hard boundaries between different parts of a workflow
- **Human-in-the-loop patterns**: When different agents need different levels of human oversight

The point isn't that multi-agent is bad—it's that it should be a conscious architectural decision, not a reflexive default.

---

## Why I'm Bullish on Single-Agent Architectures

Let me be direct about my perspective: I believe the industry has over-indexed on multi-agent systems, and this research reinforces that view.

### Complexity Without Proportional Value

Much of the multi-agent enthusiasm has been driven by impressive demos rather than production success stories. It's easy to show a swarm of agents collaborating on a task in a controlled demo. It's much harder to debug that swarm when it fails unpredictably in production.

The reality is that multi-agent coordination introduces significant complexity:
- State synchronization between agents
- Message passing and context management
- Failure handling when one agent in a chain breaks
- Observability across distributed agent contexts

For many use cases, this complexity isn't buying you proportionally better outcomes. You're paying an architectural tax without collecting sufficient benefits.

### The Debuggability Argument

When a single-agent system fails, you have one context to inspect, one decision trace to follow, one set of skill selections to analyze. When a multi-agent system fails, you're often left asking: "Which agent went wrong? What was the state when the handoff happened? Did Agent B receive what Agent A intended to send?"

This isn't a minor operational detail—debuggability directly impacts your ability to improve and maintain systems over time.

### The Cognitive Science Parallel

I find the parallel to human cognitive limits particularly compelling. Humans don't spawn sub-humans to think through complex problems. We develop skills, internalize expertise, and apply the right approach from our personal repertoire. The most capable humans aren't those running the most complex internal bureaucracies—they're those with well-developed, easily-accessible skills.

Why should we expect LLMs to be different?

### A Prediction

I expect we'll see a "simplification wave" in agent architectures over the next year or two. Teams that rushed to multi-agent systems will discover that simpler approaches often work better—and are dramatically easier to maintain. The pendulum will swing back toward single-agent architectures augmented with well-curated skills.

This doesn't mean multi-agent disappears. It means it finds its appropriate niche rather than being the hammer for every nail.

---

## The Bottom Line

Before your next agent project, I'd encourage you to start with a question: "Could a single agent with the right skills handle this?"

The research suggests that for many—perhaps most—use cases, the answer is yes. And when it is, you'll benefit from lower token costs, reduced latency, simpler debugging, and more predictable behavior.

That's not a bad trade for letting go of some architectural complexity.

For those interested in the full details, the paper "When Single-Agent with Skills Replace Multi-Agent Systems and When They Fail" is worth reading in full. It provides a rigorous foundation for what many practitioners have suspected: sometimes simpler really is better.

---

*What's your experience with single-agent vs. multi-agent architectures? I'd love to hear whether this matches what you've seen in production.*
