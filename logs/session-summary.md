{
  "sessionMetrics": {
    "models": {
      "": {
        "api": {
          "totalRequests": 20,
          "totalErrors": 1,
          "totalLatencyMs": 79170
        },
        "tokens": {
          "prompt": 1582654,
          "candidates": 4884,
          "total": 1589697,
          "cached": 1212196,
          "thoughts": 2159,
          "tool": 0
        }
      },
      "gemini-2.5-flash": {
        "api": {
          "totalRequests": 2,
          "totalErrors": 2,
          "totalLatencyMs": 2304
        },
        "tokens": {
          "prompt": 0,
          "candidates": 0,
          "total": 0,
          "cached": 0,
          "thoughts": 0,
          "tool": 0
        }
      }
    },
    "tools": {
      "totalCalls": 19,
      "totalSuccess": 18,
      "totalFail": 1,
      "totalDurationMs": 160915,
      "totalDecisions": {
        "accept": 0,
        "reject": 0,
        "modify": 0,
        "auto_accept": 19
      },
      "byName": {
        "read_file": {
          "count": 7,
          "success": 7,
          "fail": 0,
          "durationMs": 384,
          "decisions": {
            "accept": 0,
            "reject": 0,
            "modify": 0,
            "auto_accept": 7
          }
        },
        "write_file": {
          "count": 2,
          "success": 2,
          "fail": 0,
          "durationMs": 148,
          "decisions": {
            "accept": 0,
            "reject": 0,
            "modify": 0,
            "auto_accept": 2
          }
        },
        "run_shell_command": {
          "count": 5,
          "success": 5,
          "fail": 0,
          "durationMs": 159948,
          "decisions": {
            "accept": 0,
            "reject": 0,
            "modify": 0,
            "auto_accept": 5
          }
        },
        "replace": {
          "count": 5,
          "success": 4,
          "fail": 1,
          "durationMs": 435,
          "decisions": {
            "accept": 0,
            "reject": 0,
            "modify": 0,
            "auto_accept": 5
          }
        }
      }
    },
    "files": {
      "totalLinesAdded": 78,
      "totalLinesRemoved": 28
    }
  }
}