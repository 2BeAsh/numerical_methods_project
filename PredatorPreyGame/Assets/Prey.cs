using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Prey : MonoBehaviour
{
    Animator animator;
    public AudioSource audioPlayerCollide;
    public AudioSource audioPlayerDeath;

    public float Health
    {
        set
        {
            health = value;
            if (health <= 0)
            {
                Defeated();
            }
        }
        get
        {
            return health;
        }

    }

    public float health = 1;

    public void Start()
    {
        animator = GetComponent<Animator>();
    }


    public void Defeated()
    {
        audioPlayerDeath.Play();
        animator.SetTrigger("Defeated");

    }

    public void RemoveEnemy()
    {

        Destroy(gameObject);
    }

    public void OnCollisionEnter2D(Collision2D collision)
    {
        if(collision.gameObject.tag == "Player")
        {
            audioPlayerCollide.Play();
        }
    }


}
