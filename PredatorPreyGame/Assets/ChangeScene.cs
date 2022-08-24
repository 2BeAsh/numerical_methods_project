using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class ChangeScene : MonoBehaviour
{

    public void ExitButton()
    {
        Debug.Log("Game closed");
        Application.Quit();
            
    }

    public void StartGame()
    {
        SceneManager.LoadScene("Game");
    }


}
