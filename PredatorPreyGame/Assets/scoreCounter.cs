using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class scoreCounter : MonoBehaviour
{
    public static scoreCounter instance;

    public TMP_Text scoreText;
    public static int currentScore;


    private void Awake()
    {
        instance = this;
    }

    // Start is called before the first frame update
    void Start()
    {
        currentScore = 0;
        scoreText.text = "SCORE: " + currentScore.ToString();
     }


    public void increaseScore(int value)
    {
        currentScore += value;
        scoreText.text = "SCORE: " + currentScore.ToString();
    }
}
