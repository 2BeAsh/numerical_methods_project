using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class MainMenu : MonoBehaviour
{
 public void ExitGame()
    {
        Application.Quit();
        Debug.Log("Game Closed");
    }

    public void LoadMainMenu()
    {
        SceneManager.LoadScene("MainMenu");
    }

    public void CharacterSelection()
    {
        SceneManager.LoadScene("CharacterSelection");
    }

    public void StartGame()
    {
        SceneManager.LoadScene("Game");
    }
}
