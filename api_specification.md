# Tf Openpose
 
## endpoint [/]

### tf-openpose [POST]
 
#### 処理概要

* Image画像を受け取り、人間の関節の座標を返却する
 
+ Request (image/jpeg)
 
    + Headers
 
            Content-Type: image/jpeg

    + Parameters

        Image file
 
+ Response 200 (application/json)
 
    + Parameters

      |Json Key|型|必須|説明|
      |:--|:--:|:--:|:--:|
      |humans|配列| | |
      |&nbsp;&nbsp;parts|配列| | |
      ||オブジェクト| | |
      |&nbsp;&nbsp;&nbsp;&nbsp;id|数値|| |
      |&nbsp;&nbsp;&nbsp;&nbsp;score|数値|| |
      |&nbsp;&nbsp;&nbsp;&nbsp;x|数値|| |
      |&nbsp;&nbsp;&nbsp;&nbsp;y|数値|| |


    + example
      ```
        {
          "parts": [
            {
              "id": 1,
              "score": 8.773357514765799,
              "x": 0.5185185185185185,
              "y": 0.19565217391304346
            },
            {
              "id": 2,
              "score": 7.201893736489372,
              "x": 0.42592592592592593,
              "y": 0.19565217391304346
            }
          ]
        }
      ```

