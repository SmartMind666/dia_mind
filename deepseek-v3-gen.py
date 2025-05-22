import asyncio
import argparse
import logging
import random
import openai
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
import os

log = logging.getLogger()

llm_prompt_templates = {
    "deepseek-v3 output": ChatPromptTemplate([("human", "{user_input}")]),
}

async def llm_invoke(llm, task):
    try:
        message = llm_prompt_templates[task["column"]].invoke({"user_input", task["input"]})
        response = await llm.ainvoke(message)
        return {"status": "successed", "response": response.content}
    except openai.BadRequestError as e:
        log.warning(f"Task ({task["row"]}, {task["column"]}) has illegal contents, skipped")
        return {"status": "successed", "response": ""}
    except BaseException:
        return {"status": "failed"}

async def task_executor(llm, task):
    retries = 0
    while True:
        response = await llm_invoke(llm, task)
        if response["status"] == "successed":
            return {"task": task, "response": response["response"], "retries": retries}
        log.warning(f"Task ({task["row"]}, {task["column"]}) failed, retrying")
        retries += 1
        await asyncio.sleep(random.randint(10, 60))
            
async def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s][%(name)s]%(message)s")

    parse = argparse.ArgumentParser()
    parse.add_argument("file", type=str)
    parse.add_argument("-k", "--api-key", type=str)
    parse.add_argument("-c", "--max-concurrent", default=10, type=int)
    parse.add_argument("-e", "--endurance", default=5, type=int)
    args = parse.parse_args()

    log.info(f"Loading data from \"{args.file}\"")
    df = pd.read_excel(args.file, sheet_name="Sheet1")
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    if not "deepseek-v3 output" in df.columns:
        df.insert(df.shape[1], "deepseek-v3 output", None)
    df = df.astype({"deepseek-v3 output": "object"})
    
    # Find empty cell
    log.info("Generating task list")
    indices = df[df["deepseek-v3 output"].isna()].index.tolist()
    task_list = list(map(lambda i: {"row": i, "column": "deepseek-v3 output", "input": df.at[i, "input"]}, indices))
    log.info(f"{len(task_list)} tasks recorded")

    # Config LLM
    llm = ChatDeepSeek(
        model="deepseek-chat",
        streaming=False,
        api_key=args.api_key,
    )

    # Process tasks
    log.info("Start to processing tasks")
    concurrent = args.max_concurrent
    consecutive_no_retries = 0
    coroutine_list = []
    for i, task in enumerate(task_list):
        coroutine_list.append(task_executor(llm, task))
        if len(coroutine_list) == concurrent or i == (len(task_list) - 1):
            responses = await asyncio.gather(*coroutine_list)
            for response in responses:
                df.at[response["task"]["row"], response["task"]["column"]] = response["response"]
            # Backup
            if os.path.exists(f"{args.file}.bak"):
                os.rename(f"{args.file}.bak", f"_{args.file}.bak")
            os.rename(args.file, f"{args.file}.bak")
            if os.path.exists(f"_{args.file}.bak"):
                os.remove(f"_{args.file}.bak")
            # Save results
            df.to_excel(args.file, index=False)
            log.info(f"Finished {i + 1} tasks of {len(task_list)}")
            # Cleanup coroutine list
            coroutine_list.clear()
            # Update concurrent
            retries = sum([x["retries"] for x in responses])
            if retries == 0:
                consecutive_no_retries += 1
                if consecutive_no_retries == args.endurance:
                    concurrent = min(args.max_concurrent, concurrent + 1)
                    consecutive_no_retries = 0
                    log.info(f"Concurrent updated to {concurrent}")
            else:
                concurrent = max(1, concurrent - 1)
                consecutive_no_retries = 0
                log.info(f"Concurrent updated to {concurrent}")

    log.info("All tasks finished")

asyncio.run(main())
