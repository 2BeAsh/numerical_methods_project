using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Prey : MonoBehaviour
{
    Animator animator;
    public AudioSource audioPlayerCollide;
    public AudioSource audioPlayerDeath;
    
    /*
    public ParticleSystem deathparticlesystem;
    public SpriteRenderer sr;

    public bool once = true; 
    */
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
    public int scoreIncreaseAmount;

    public void Start()
    {
        animator = GetComponent<Animator>();
        //deathparticlesystem = GetComponent<ParticleSystem>();
    }


    public void Defeated()
    {
        scoreCounter.instance.increaseScore(scoreIncreaseAmount);
        audioPlayerDeath.Play();
        
            
        animator.SetTrigger("Defeated");

    }

    public void RemoveEnemy()
    {
        /*
        var em = deathparticlesystem.emission;
        var dur = deathparticlesystem.main.duration;

        em.enabled = true;
        deathparticlesystem.Play();
        once = false;
        //Destroy(sr);
        Invoke(nameof(DestroyObject), dur);
        */
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
