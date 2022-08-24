using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using TMPro;

public class GameOverScreen : MonoBehaviour
{

    public TMP_Text finalScore;


    public void Setup(int score)
    {
        gameObject.SetActive(true);
        finalScore.text = score.ToString() + " POINTS";
    }


    public void RestartButton()
    {
        SceneManager.LoadScene("Game");
    }

    public void ExitButton() {
        SceneManager.LoadScene("MainMenu");
    }

}
