{
  "sessionMetrics": {
    "models": {
      "": {
        "api": {
          "totalRequests": 11,
          "totalErrors": 0,
          "totalLatencyMs": 54684
        },
        "tokens": {
          "prompt": 182582,
          "candidates": 532,
          "total": 184960,
          "cached": 125756,
          "thoughts": 1846,
          "tool": 0
        }
      }
    },
    "tools": {
      "totalCalls": 7,
      "totalSuccess": 7,
      "totalFail": 0,
      "totalDurationMs": 4854,
      "totalDecisions": {
        "accept": 0,
        "reject": 0,
        "modify": 0,
        "auto_accept": 7
      },
      "byName": {
        "read_file": {
          "count": 1,
          "success": 1,
          "fail": 0,
          "durationMs": 45,
          "decisions": {
            "accept": 0,
            "reject": 0,
            "modify": 0,
            "auto_accept": 1
          }
        },
        "run_shell_command": {
          "count": 2,
          "success": 2,
          "fail": 0,
          "durationMs": 4606,
          "decisions": {
            "accept": 0,
            "reject": 0,
            "modify": 0,
            "auto_accept": 2
          }
        },
        "glob": {
          "count": 1,
          "success": 1,
          "fail": 0,
          "durationMs": 78,
          "decisions": {
            "accept": 0,
            "reject": 0,
            "modify": 0,
            "auto_accept": 1
          }
        },
        "write_file": {
          "count": 3,
          "success": 3,
          "fail": 0,
          "durationMs": 125,
          "decisions": {
            "accept": 0,
            "reject": 0,
            "modify": 0,
            "auto_accept": 3
          }
        }
      }
    },
    "files": {
      "totalLinesAdded": 16,
      "totalLinesRemoved": 5
    }
  }
}