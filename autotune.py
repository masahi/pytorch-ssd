import numpy as np
import onnx
import tvm
from tvm import relay, auto_scheduler
from tvm.runtime.vm import VirtualMachine


model = onnx.load("mb1-ssd.onnx")

ishape = (1, 3, 300, 300)
shape_dict = {"input.1": ishape}
target = "vulkan"
log_file = "logs/ssd-mb1-vulkan.log"


def auto_schedule(mod, params):
    tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target)

    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=1000)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    # tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=100000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)


# mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)
# mod = relay.transform.DynamicToStatic()(mod)

# with open("ssd-mb1_mod.json", "w") as fo:
#     fo.write(tvm.ir.save_json(mod))
# with open("ssd-mb1.params", "wb") as fo:
#     fo.write(relay.save_param_dict(params))

with open("ssd-mb1_mod.json", "r") as fi:
    mod = tvm.ir.load_json(fi.read())
with open("ssd-mb1.params", "rb") as fi:
    params = relay.load_param_dict(fi.read())

auto_schedule(mod, params)

inp = np.random.randn(1, 3, 300, 300)

with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        vm_exec = relay.vm.compile(mod, target=target, params=params)

ctx = tvm.context(target, 0)
vm = VirtualMachine(vm_exec, ctx)
vm.set_input("main", **{"input.1": inp})
vm.run()

ftimer = vm.module.time_evaluator("invoke", ctx, number=1, repeat=num_iters)
print(ftimer("main"))
