import io

from weedseg.console import LineLog, Tee


def test_log_gets_the_final_state_of_redrawn_lines():
    console, log = io.StringIO(), io.StringIO()
    line_log = LineLog(log)
    out, err = Tee(console, line_log), Tee(io.StringIO(), line_log)
    out.write("Run x, ")
    out.write("device cpu\n")
    for done in range(4):
        err.write(f"\repoch 0: {done}/3")  # progress bars go to stderr
    out.write("\nDone\n")
    err.write("\rpredict: 1/9")
    line_log.close_line()  # e.g. the step ended mid-bar
    assert console.getvalue() == "Run x, device cpu\n\nDone\n"
    assert log.getvalue() == "Run x, device cpu\nepoch 0: 3/3\nDone\npredict: 1/9\n"
